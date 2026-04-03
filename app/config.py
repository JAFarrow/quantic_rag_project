from __future__ import annotations

import os
from dataclasses import dataclass

APP_VERSION = "0.1.0"


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Received '{raw_value}'.") from exc


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float. Received '{raw_value}'.") from exc


@dataclass(frozen=True)
class ChatSettings:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    openai_chat_model: str
    openai_embedding_model: str
    chat_top_k: int
    chat_min_score: float

    @classmethod
    def from_env(cls) -> "ChatSettings":
        return cls(
            openai_api_key=_require("OPENAI_API_KEY"),
            pinecone_api_key=_require("PINECONE_API_KEY"),
            pinecone_index_name=_require("PINECONE_INDEX_NAME"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "dev"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_top_k=_read_int("CHAT_TOP_K", 4),
            chat_min_score=_read_float("CHAT_MIN_SCORE", 0.4),
        )


__all__ = ["APP_VERSION", "ChatSettings"]
