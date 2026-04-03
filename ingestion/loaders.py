from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def discover_pdf_paths(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.glob("*.pdf") if path.is_file())


def load_documents(pdf_paths: list[Path]) -> list[Document]:
    documents: list[Document] = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        loaded_documents = loader.load()

        for document in loaded_documents:
            document.metadata["source"] = str(pdf_path)
            document.metadata["filename"] = pdf_path.name
            documents.append(document)

    return documents
