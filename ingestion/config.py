from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Received '{raw_value}'.") from exc


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class IngestionSettings:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    embedding_model: str
    data_dir: Path
    chunk_size: int
    chunk_overlap: int
    batch_size: int

    @classmethod
    def from_env(cls) -> "IngestionSettings":
        return cls(
            openai_api_key=_require("OPENAI_API_KEY"),
            pinecone_api_key=_require("PINECONE_API_KEY"),
            pinecone_index_name=_require("PINECONE_INDEX_NAME"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            data_dir=Path(os.getenv("INGESTION_DATA_DIR", "data")),
            chunk_size=_read_int("INGESTION_CHUNK_SIZE", 1000),
            chunk_overlap=_read_int("INGESTION_CHUNK_OVERLAP", 200),
            batch_size=_read_int("INGESTION_BATCH_SIZE", 100),
        )
