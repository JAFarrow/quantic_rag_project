from __future__ import annotations

from dataclasses import dataclass

from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ingestion.chunking import chunk_documents
from ingestion.config import IngestionSettings
from ingestion.loaders import discover_pdf_paths, load_documents


@dataclass(frozen=True)
class IngestionResult:
    pdf_count: int
    document_count: int
    chunk_count: int
    upserted_count: int
    namespace: str
    index_name: str


def _discover_inputs(settings: IngestionSettings) -> dict[str, list]:
    pdf_paths = discover_pdf_paths(settings.data_dir)
    if not pdf_paths:
        raise ValueError(f"No PDF files found in directory: {settings.data_dir}")
    return {"pdf_paths": pdf_paths}


def _load_input_documents(payload: dict[str, list]) -> dict[str, list]:
    payload["documents"] = load_documents(payload["pdf_paths"])
    return payload


def _chunk_input_documents(settings: IngestionSettings, payload: dict[str, list]) -> dict[str, list]:
    chunks = chunk_documents(
        documents=payload["documents"],
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError("No text chunks were produced from the provided documents.")
    payload["chunks"] = chunks
    return payload


def ingest(settings: IngestionSettings) -> IngestionResult:
    chunking_chain = (
        RunnableLambda(lambda _: _discover_inputs(settings))
        | RunnableLambda(_load_input_documents)
        | RunnableLambda(lambda payload: _chunk_input_documents(settings, payload))
    )
    payload = chunking_chain.invoke(None)

    embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

    pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    index = pinecone_client.Index(settings.pinecone_index_name)
    index.delete(delete_all=True, namespace=settings.pinecone_namespace)

    PineconeVectorStore.from_documents(
        documents=payload["chunks"],
        embedding=embeddings,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        pinecone_api_key=settings.pinecone_api_key,
    )

    return IngestionResult(
        pdf_count=len(payload["pdf_paths"]),
        document_count=len(payload["documents"]),
        chunk_count=len(payload["chunks"]),
        upserted_count=len(payload["chunks"]),
        namespace=settings.pinecone_namespace,
        index_name=settings.pinecone_index_name,
    )
