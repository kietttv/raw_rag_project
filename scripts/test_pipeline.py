"""End-to-End verification script for RAG Core Pipeline.

Workflow:
  1. Ingest sample document (data/samples/sample.txt) into ChromaDB.
  2. Verify vectors stored in ChromaDB.
  3. Query ChromaDB to retrieve top relevant chunks.
  4. Generate answer with Gemini (gemini-3.5-flash).
  5. Print results and validate output.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.core.generator import generate_rag_response
from src.core.ingestion import ingest_path
from src.services.vector_db import get_chroma_client, get_collection
from src.utils.logger import logger


def run_test_pipeline(
    sample_file: Path | str = BASE_DIR / "data" / "samples" / "sample.txt",
    collection_name: str = "rag_test_pipeline",
    question: str = "Lợi ích của RAG là gì?",
) -> None:
    sample_path = Path(sample_file).resolve()
    print("\n" + "=" * 70)
    print(">>> STARTING INDEPENDENT RAG PIPELINE TEST")
    print("=" * 70)
    print(f"Sample File:     {sample_path}")
    print(f"Collection Name: {collection_name}")
    print(f"Question:        {question}")
    print("=" * 70)

    # 1. Check ChromaDB server connectivity
    print("\n[Step 1/4] Checking ChromaDB connection...")
    try:
        client = get_chroma_client()
        client.heartbeat()
        print("  --> [OK] Connected to ChromaDB server successfully.")
    except Exception as exc:
        print(f"  --> [FAIL] Could not connect to ChromaDB at {settings.VECTOR_DB_URL}: {exc}")
        print("       Please ensure ChromaDB server is running:")
        print("       chroma run --path ./chroma_data --port 8000")
        sys.exit(1)

    # 2. Ingest document
    print(f"\n[Step 2/4] Ingesting document: {sample_path.name}...")
    if not sample_path.exists():
        print(f"  --> [FAIL] Sample file not found at {sample_path}")
        sys.exit(1)

    ingest_result = ingest_path(sample_path, collection_name=collection_name)
    print(f"  --> [OK] Files processed: {ingest_result.files_processed}")
    print(f"  --> [OK] Chunks created/upserted: {ingest_result.chunks_created}")

    # Verify count in Chroma
    col = get_collection(collection_name)
    total_docs = col.count()
    print(f"  --> [OK] Total items in collection '{collection_name}': {total_docs}")
    assert total_docs > 0, "Collection should have at least 1 document after ingestion"

    # 3 & 4. Retrieve & Generate Answer
    print(f"\n[Step 3/4] Retrieving context & [Step 4/4] Generating answer with Gemini...")
    response = generate_rag_response(
        query=question,
        top_k=3,
        collection_name=collection_name,
    )

    print("\n" + "=" * 70)
    print("RAG PIPELINE EXECUTION RESULT")
    print("=" * 70)
    print(f"QUESTION: {question}\n")
    print("ANSWER:")
    print(response.answer)
    print("\n" + "-" * 70)
    print(f"SOURCES CITED ({len(response.sources)} retrieved chunks):")
    print("-" * 70)
    for idx, doc in enumerate(response.sources, start=1):
        score_str = f"{doc.score:.4f}" if doc.score is not None else "N/A"
        source = doc.metadata.get("file_name") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        page_str = f" (Page {page})" if page else ""
        print(f"[{idx}] Score: {score_str} | Source: {source}{page_str}")
        print(f"    Content snippet: {doc.content.strip()[:180]}...\n")

    # Final Validation Assertions
    assert len(response.answer.strip()) > 0, "Generated answer should not be empty"
    assert len(response.sources) > 0, "Should have retrieved at least 1 source chunk"
    print("=" * 70)
    print(">>> [SUCCESS] END-TO-END RAG PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_test_pipeline()

