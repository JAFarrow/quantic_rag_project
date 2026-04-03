from __future__ import annotations

import re

from langchain_core.documents import Document

SYSTEM_PROMPT = """
You are a policy question-answering assistant.

You must answer using only the provided context.
Do not use outside knowledge.
Do not guess, infer, or invent policy details that are not explicitly supported by the context.

Requirements:
1. Give a concise, accurate answer to the user's question.
2. If the context is insufficient, say that the answer is not available in the provided documents.
3. If the provided context appears to conflict, briefly note that the documents conflict.
4. Add citation markers to every factual sentence using the provided context ids, formatted as [n].

Output rules:
- Write in plain language.
- Keep the answer short unless the question clearly requires a list.
- Use only citation ids that appear in the context.
- Do not include a bibliography or source list in the answer.
""".strip()

USER_PROMPT_TEMPLATE = """Answer the question using only the context below.

Question:
{question}

Context:
{context}

Return only the answer.
Every factual sentence must include one or more citation markers like [1] or [2][3].
Use only citation ids from the context above."""

CITATION_RETRY_PROMPT_TEMPLATE = """Answer the question using only the context below.

Question:
{question}

Context:
{context}

Draft answer missing citations:
{draft_answer}

Rewrite the answer and include citation markers like [1] in every factual sentence.
Return only the answer.
Use only citation ids from the context above."""


def build_citations(documents: list[Document]) -> list[dict[str, int | str | None]]:
    citations: list[dict[str, int | str | None]] = []

    for citation_id, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source = str(metadata.get("filename") or metadata.get("source") or "unknown")
        page = metadata.get("page_label") or metadata.get("page")

        citations.append(
            {
                "id": citation_id,
                "source": source,
                "page": page,
                "content": document.page_content.strip(),
            }
        )

    return citations


def build_context(citations: list[dict[str, int | str | None]]) -> str:
    context_sections: list[str] = []

    for citation in citations:
        content = str(citation.get("content") or "").strip()
        if content:
            source = str(citation.get("source") or "unknown")
            page = citation.get("page")
            page_text = str(page) if page is not None else "unknown"
            context_sections.append(
                f"[{citation['id']}] Source: {source}; Page: {page_text}\n{content}"
            )

    return "\n\n".join(context_sections)


def build_chain_payload(inputs: dict[str, object]) -> dict[str, object]:
    question = str(inputs.get("question") or "").strip()
    documents = list(inputs.get("documents") or [])
    citations = build_citations([doc for doc in documents if isinstance(doc, Document)])
    context = build_context(citations)
    return {
        "question": question,
        "context": context,
        "citations": citations,
    }


def build_retry_payload(inputs: dict[str, object]) -> dict[str, object]:
    return {
        "question": str(inputs.get("question") or "").strip(),
        "context": str(inputs.get("context") or "").strip(),
        "draft_answer": str(inputs.get("draft_answer") or "").strip(),
    }


def extract_citation_ids(answer: str) -> set[int]:
    return {int(match) for match in re.findall(r"\[(\d+)\]", answer)}


def filter_citations_by_answer(
    answer: str,
    citations: list[dict[str, int | str | None]],
) -> list[dict[str, int | str | None]]:
    used_ids = extract_citation_ids(answer)
    if not used_ids:
        return []

    return [citation for citation in citations if int(citation["id"]) in used_ids]


def serialize_citations(citations: list[dict[str, int | str | None]]) -> list[dict[str, int | str | None]]:
    return [
        {
            "id": int(citation["id"]),
            "source": str(citation["source"]),
            "page": citation.get("page"),
        }
        for citation in citations
    ]
