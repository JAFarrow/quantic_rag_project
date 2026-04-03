from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.config import ChatSettings
from app.services.chat_helpers import (
    CITATION_RETRY_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_chain_payload,
    build_retry_payload,
    extract_citation_ids,
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

    citation_retry_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", CITATION_RETRY_PROMPT_TEMPLATE),
        ]
    )

    citation_retry_chain = (
        RunnableLambda(build_retry_payload)
        | citation_retry_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, citation_retry_chain


def answer_question(
    question: str,
    settings: ChatSettings,
) -> tuple[str, list[dict[str, int | str | None]]]:
    rag_chain, citation_retry_chain = _build_rag_chain(settings)

    chain_result = rag_chain.invoke(question)
    citations = list(chain_result.get("citations") or [])
    answer = str(chain_result.get("answer") or "").strip()

    if not citations:
        return (
            "I couldn't find enough relevant information in the provided policy documents to answer that question.",
            [],
        )

    if citations and not extract_citation_ids(answer):
        answer = citation_retry_chain.invoke(
            {
                "question": question,
                "context": str(chain_result.get("context") or ""),
                "draft_answer": answer,
            }
        )

    cited_chunks = filter_citations_by_answer(answer, citations)
    serialized_citations = serialize_citations(cited_chunks)

    return answer, serialized_citations
