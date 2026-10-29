"""Utility script to clear or reset ChromaDB collections and in-memory caches."""

# Show help
# python scripts/clear_db.py --help

# Clear default collection
# python scripts/clear_db.py

# Clear specific collection
# python scripts/clear_db.py -c custom_collection

# Delete all collections
# python scripts/clear_db.py -a

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.core.retriever import reset_bm25_cache
from src.services import cache_service
from src.services.vector_db import get_chroma_client, get_collection
from src.utils.logger import logger


def clear_collection(name: str) -> None:
    """Delete and recreate a specific collection, invalidating caches."""
    client = get_chroma_client()
    try:
        col = client.get_collection(name)
        count = col.count()
        client.delete_collection(name)

        # Re-create empty collection with default settings
        get_collection(name)

        # Invalidate BM25 and Redis caches
        reset_bm25_cache()
        cache_service.invalidate_collection_cache(name)

        logger.bind(collection=name, count=count).info("Successfully cleared collection")
        print(f"[OK] Successfully cleared collection '{name}' ({count} vectors).")
    except Exception as exc:
        logger.bind(collection=name, error=str(exc)).error("Failed to clear collection")
        print(f"[WARN] Failed to clear collection '{name}': {exc}")


def clear_all_collections() -> None:
    """Delete all collections in ChromaDB and recreate default collection."""
    client = get_chroma_client()
    cols = client.list_collections()
    if not cols:
        print("[INFO] ChromaDB is currently empty; no collections found.")
        return

    for col in cols:
        name = col.name
        count = col.count()
        client.delete_collection(name)
        cache_service.invalidate_collection_cache(name)
        print(f"[OK] Deleted collection '{name}' ({count} vectors).")

    # Re-create default collection
    default_name = settings.CHROMA_COLLECTION_NAME
    get_collection(default_name)
    reset_bm25_cache()
    logger.info("Reset all ChromaDB collections and recreated default collection")
    print(f"[OK] Successfully reset all ChromaDB collections and recreated default collection '{default_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB Database Clear & Reset Utility")
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        help="Name of the target collection to clear (defaults to CHROMA_COLLECTION_NAME from settings)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Delete all collections across ChromaDB",
    )

    args = parser.parse_args()

    if args.all:
        confirm = input("Are you sure you want to delete ALL collections? (y/N): ").strip().lower()
        if confirm == "y":
            clear_all_collections()
        else:
            print("[INFO] Operation cancelled by user.")
    elif args.collection:
        clear_collection(args.collection)
    else:
        target = settings.CHROMA_COLLECTION_NAME
        print(f"[INFO] Clearing default collection: '{target}'...")
        clear_collection(target)
