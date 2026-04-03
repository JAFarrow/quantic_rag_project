from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.config import ChatSettings


SYSTEM_PROMPT = """
You are a policy question-answering assistant.

You must answer using only the provided context.
Do not use outside knowledge.
Do not guess, infer, or invent policy details that are not explicitly supported by the context.

Requirements:
1. Give a concise, accurate answer to the user's question.
2. If the context is insufficient, say that the answer is not available in the provided documents.
3. If the provided context appears to conflict, briefly note that the documents conflict.

Output rules:
- Write in plain language.
- Keep the answer short unless the question clearly requires a list.
- Do not include citations, source names, page numbers, or reference markers in the answer.
""".strip()


def _build_context(documents: list[Document]) -> str:
    context_sections: list[str] = []

    for document in documents:
        content = document.page_content.strip()
        if content:
            context_sections.append(content)

    return "\n\n".join(context_sections)


def _extract_sources(documents: list[Document]) -> list[dict[str, int | str | None]]:
    seen: set[tuple[str, str | int | None]] = set()
    sources: list[dict[str, int | str | None]] = []

    for document in documents:
        metadata = document.metadata or {}
        source = str(metadata.get("filename") or metadata.get("source") or "unknown")
        page = metadata.get("page_label") or metadata.get("page")

        key = (source, page)
        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "source": source,
                "page": page,
            }
        )

    return sources


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(part for part in text_parts if part).strip()

    return str(content).strip()


def _build_user_prompt(question: str, context: str) -> str:
    return f"""Answer the question using only the context below.

Question:
{question}

Context:
{context}

Return only the answer."""


def answer_question(question: str, settings: ChatSettings) -> tuple[str, list[dict[str, int | str | None]]]:
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )

    vector_store = PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
        namespace=settings.pinecone_namespace,
    )

    results = vector_store.similarity_search_with_score(
        query=question,
        k=settings.chat_top_k,
    )

    if not results:
        return (
            "I couldn't find enough information in the provided policy documents to answer that question.",
            [],
        )

    MIN_SCORE = settings.chat_min_score
    filtered_results = [(doc, score) for doc, score in results if score >= MIN_SCORE]
    documents = [doc for doc, _score in filtered_results]

    if not documents:
        return (
            "I couldn't find enough relevant information in the provided policy documents to answer that question.",
            [],
        )

    sources = _extract_sources(documents)
    context = _build_context(documents)

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(question, context)),
        ]
    )

    answer = _message_content_to_text(response.content)
    return answer, sources