"""CLI script to run full ingestion and vector indexing pipeline."""

import argparse
import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import BASE_DIR, setup_logger
from src.indexing.vector_store import get_vector_store, index_documents
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import load_directory

logger = setup_logger("scripts.ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest medical documents and build Chroma index.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(BASE_DIR / "data" / "raw"),
        help="Path to directory containing medical files",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing vector collection before indexing",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error("Data directory does not exist: %s", data_path)
        sys.exit(1)

    logger.info("Loading documents from %s...", data_path)
    raw_docs = load_directory(data_path)
    if not raw_docs:
        logger.warning("No documents found in %s.", data_path)
        return

    logger.info("Chunking %d raw documents...", len(raw_docs))
    chunks = chunk_documents(raw_docs)

    vector_store = get_vector_store()
    if args.reset:
        logger.info("Resetting existing vector store collection...")
        vector_store.reset_collection()

    logger.info("Indexing %d chunks into Chroma DB...", len(chunks))
    indexed_count = index_documents(chunks, vector_store=vector_store)
    logger.info("Ingestion completed successfully: %d chunks ready for retrieval.", indexed_count)


if __name__ == "__main__":
    main()
