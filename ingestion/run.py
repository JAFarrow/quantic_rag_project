from __future__ import annotations

from dotenv import load_dotenv

from ingestion.config import IngestionSettings
from ingestion.pipeline import ingest


def main() -> None:
    load_dotenv()
    settings = IngestionSettings.from_env()

    print(f"Starting ingestion for index '{settings.pinecone_index_name}' namespace '{settings.pinecone_namespace}'")
    result = ingest(settings)
    print(
        "Ingestion complete: "
        f"pdfs={result.pdf_count}, documents={result.document_count}, "
        f"chunks={result.chunk_count}, upserted={result.upserted_count}"
    )


if __name__ == "__main__":
    main()
