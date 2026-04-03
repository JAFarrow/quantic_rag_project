from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: list[Document] = []
    for doc_index, document in enumerate(documents):
        split_chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(split_chunks):
            chunk.metadata["doc_index"] = doc_index
            chunk.metadata["chunk_index"] = chunk_index

            source = chunk.metadata.get("source", f"doc-{doc_index}")
            page = chunk.metadata.get("page", "unknown")
            chunk.metadata["chunk_id"] = f"{source}-p{page}-c{chunk_index}"

        chunks.extend(split_chunks)

    return chunks