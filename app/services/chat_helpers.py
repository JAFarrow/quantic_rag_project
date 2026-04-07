from __future__ import annotations

import re

from langchain_core.documents import Document

SYSTEM_PROMPT = """
You are a policy question-answering assistant.

You must answer using only the provided context.
Do not use outside knowledge.
Do not guess, infer, generalize, or invent policy details that are not explicitly supported by the context.

Priority rules:
1. Prefer direct, specific policy language over broad summaries.
2. If a local clause, exception, condition, or qualifier is present, include it rather than replacing it with a general statement.
3. Do not combine information across different documents unless the question explicitly requires cross-document comparison or synthesis.
4. Do not claim that the policy does not mention, does not specify, or does not explicitly state something unless that conclusion is clearly supported by the provided context.
5. If the context is insufficient to answer any part of the question, say so plainly.
6. If the provided context appears to conflict, briefly note the conflict.
7. Add citation markers to factual statements that are supported by the provided context ids, formatted as [n].
8. If the context is insufficient to answer the question, return this exact sentence with no citation markers: "I couldn't find enough relevant information in the provided policy documents to answer that question."

Output rules:
- Write in plain language.
- Keep the answer short unless the question clearly requires a list.
- Use only citation ids that appear in the context.
- Do not include a bibliography or source list.
- Do not cite a sentence unless the cited passage directly supports that sentence.
""".strip()

USER_PROMPT_TEMPLATE = """Answer the question using only the context below.

Question:
{question}

Context:
{context}

Before writing the final answer, identify:
- the exact facts needed to answer the question
- which citation ids support each fact

Then write the final answer using only those supported facts.

Rules:
- Answer every part of the question.
- Prefer specific clauses over general summaries.
- Do not add unsupported details.
- Do not merge information across documents unless the question explicitly requires it.
- If the context is insufficient for any part, say that plainly.
- Every supported factual sentence should include one or more citation markers like [1] or [2][3].
- If the context is insufficient, return exactly: "I couldn't find enough relevant information in the provided policy documents to answer that question." and do not include citation markers.
- Use only citation ids from the context above.

Return only the final answer."""

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


def filter_citations_by_answer(
    answer: str,
    citations: list[dict[str, int | str | None]],
) -> list[dict[str, int | str | None]]:
    used_ids = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
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
