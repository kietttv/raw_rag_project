"""Ingestion endpoints for uploading and processing documents asynchronously."""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from src.config import BASE_DIR, settings
from src.core.ingestion import SUPPORTED_EXTENSIONS, ingest_path
from src.models.schemas import (
    IngestTaskDetail,
    IngestTaskResponse,
    TaskStatus,
)
from src.services.task_store import task_store
from src.utils.logger import logger

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _process_ingestion(
    task_id: str,
    file_path: Path,
    target_collection: str,
    task_dir: Path,
) -> None:
    """Synchronous background worker executed in a threadpool."""
    logger.bind(task_id=task_id, path=str(file_path), collection=target_collection).info(
        "Starting background ingestion task"
    )
    task_store.update_task(task_id=task_id, status=TaskStatus.PROCESSING)

    try:
        result = ingest_path(path=file_path, collection_name=target_collection)

        if result.failed_files and result.chunks_created == 0:
            err_msg = f"Failed to ingest file(s): {', '.join(result.failed_files)}"
            logger.bind(task_id=task_id, error=err_msg).error("Ingestion task failed")
            task_store.update_task(
                task_id=task_id,
                status=TaskStatus.FAILED,
                failed_files=result.failed_files,
                error=err_msg,
            )
        else:
            logger.bind(
                task_id=task_id,
                files=result.files_processed,
                chunks=result.chunks_created,
            ).info("Background ingestion task completed")
            task_store.update_task(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                files_processed=result.files_processed,
                chunks_created=result.chunks_created,
                failed_files=result.failed_files,
            )
    except Exception as exc:
        err_msg = str(exc)
        logger.bind(task_id=task_id).exception(f"Unhandled exception during ingestion: {exc}")
        task_store.update_task(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=err_msg,
        )
    finally:
        # Clean up temporary task upload directory
        try:
            if task_dir.exists():
                shutil.rmtree(task_dir, ignore_errors=True)
                logger.bind(task_id=task_id, dir=str(task_dir)).debug("Cleaned up upload task directory")
        except Exception as cleanup_err:
            logger.bind(task_id=task_id, error=str(cleanup_err)).warning(
                "Failed to clean up temporary upload directory"
            )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestTaskResponse,
    summary="Upload and ingest a document asynchronously",
)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (.pdf or .txt)"),
    collection_name: str | None = Form(
        None,
        description="Target ChromaDB collection name (defaults to configured CHROMA_COLLECTION_NAME)",
    ),
) -> IngestTaskResponse:
    """Upload a single file (.pdf or .txt) to be parsed, chunked, embedded, and stored."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename",
        )

    # 1. Validate file extension
    raw_name = Path(file.filename).name  # Sanitize against path traversal
    ext = Path(raw_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{ext}'. Allowed extensions are: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # 2. Determine target collection
    target_collection = (
        collection_name.strip()
        if (collection_name and collection_name.strip())
        else settings.CHROMA_COLLECTION_NAME
    )

    # 3. Create unique task ID and temporary directory
    task_id = uuid.uuid4().hex
    upload_root = BASE_DIR / settings.UPLOAD_DIR
    task_dir = upload_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    destination = task_dir / raw_name

    # 4. Stream file to disk while enforcing max file size
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    total_bytes = 0
    chunk_buffer_size = 1024 * 1024  # 1MB buffer

    try:
        with open(destination, "wb") as out_file:
            while True:
                chunk = await file.read(chunk_buffer_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    out_file.close()
                    shutil.rmtree(task_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            f"File size exceeds the maximum allowed limit of "
                            f"{settings.MAX_UPLOAD_MB} MB"
                        ),
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(task_dir, ignore_errors=True)
        logger.bind(filename=raw_name).exception(f"Error saving uploaded file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file",
        )
    finally:
        await file.close()

    # If file is 0 bytes, reject it
    if total_bytes == 0:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)",
        )

    # 5. Register task in store
    task_store.create_task(
        task_id=task_id,
        file_name=raw_name,
        collection_name=target_collection,
    )

    # 6. Schedule background ingestion task
    background_tasks.add_task(
        _process_ingestion,
        task_id=task_id,
        file_path=destination,
        target_collection=target_collection,
        task_dir=task_dir,
    )

    logger.bind(task_id=task_id, file=raw_name, size_bytes=total_bytes).info(
        "Ingestion task queued successfully"
    )

    return IngestTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        file_name=raw_name,
        message="File uploaded successfully and scheduled for background ingestion",
    )


@router.get(
    "/{task_id}",
    response_model=IngestTaskDetail,
    summary="Get status and result of an ingestion task",
)
async def get_ingest_task(task_id: str) -> IngestTaskDetail:
    """Poll the status and processing metrics of a background ingestion task."""
    record = task_store.get_task(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found",
        )
    return record
