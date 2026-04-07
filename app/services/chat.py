from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.config import ChatSettings
from app.services.chat_helpers import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_chain_payload,
    filter_citations_by_answer,
    serialize_citations,
)


def _build_rag_chain(settings: ChatSettings):
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

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": settings.chat_top_k,
            "score_threshold": settings.chat_min_score,
        },
    )

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=settings.openai_max_output_tokens,
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE),
        ]
    )

    answer_chain = answer_prompt | llm | StrOutputParser()

    rag_chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            documents=retriever,
        )
        | RunnableLambda(build_chain_payload)
        | RunnableParallel(
            question=RunnableLambda(lambda payload: payload["question"]),
            context=RunnableLambda(lambda payload: payload["context"]),
            citations=RunnableLambda(lambda payload: payload["citations"]),
            answer=answer_chain,
        )
    )

    return rag_chain


def answer_question(
    question: str,
    settings: ChatSettings,
) -> tuple[str, list[dict[str, int | str | None]]]:
    rag_chain = _build_rag_chain(settings)
    insufficient_context_message = (
        "I couldn't find enough relevant information in the provided policy documents to answer that question."
    )

    chain_result = rag_chain.invoke(question)
    citations = list(chain_result.get("citations") or [])
    answer = str(chain_result.get("answer") or "").strip()

    if not citations:
        return insufficient_context_message, []

    lowered_answer = answer.lower()
    if (
        "couldn't find enough relevant information" in lowered_answer
        or "not available in the provided documents" in lowered_answer
    ):
        return insufficient_context_message, []

    cited_chunks = filter_citations_by_answer(answer, citations)
    serialized_citations = serialize_citations(cited_chunks)

    return answer, serialized_citations
