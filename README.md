# quantic_rag_project

Production-style FastAPI RAG chatbot with a deployed web UI, citation-grounded answers from policy documents, and a Pinecone-backed ingestion pipeline.

## Deployed app

https://quantic-rag-project.onrender.com/

## Getting started

1.  Create and activate the project virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  Install the pinned dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Run the FastAPI server locally:

    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

## Chat API

The API includes a retrieval-augmented chat endpoint at `POST /api/chat`.

Set these environment variables before running the server:

```bash
export OPENAI_API_KEY="your-openai-key"
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_INDEX_NAME="your-index-name"
export PINECONE_NAMESPACE="dev/test/prod"
```

Optional tuning variables:

```bash
export OPENAI_CHAT_MODEL="gpt-5.4-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export CHAT_TOP_K="4"
export CHAT_MIN_SCORE="0.0"
```

Example request:

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the policy say about late fees?"}'
```

Example response:

```json
{
  "answer": "Late fees are capped at 5% of the outstanding balance [1].",
  "citations": [
    {
      "id": 1,
      "source": "policy-handbook.pdf",
      "page": 12,
    }
  ]
}
```

## Document ingestion

The repository includes a CLI ingestion pipeline that:

- parses PDFs from `data/` with LangChain
- chunks document text with `RecursiveCharacterTextSplitter`
- embeds chunks with OpenAI `text-embedding-3-small`
- wipes a Pinecone namespace, then upserts the new vectors

Set required environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_INDEX_NAME="your-index-name"
export PINECONE_NAMESPACE="dev/test/prod"
```

You can also place these in a `.env` file at the repository root. The ingestion
command loads `.env` automatically.

Run ingestion:

```bash
python -m ingestion.run
```

Adjust ingestion behavior by setting `INGESTION_DATA_DIR`, `INGESTION_CHUNK_SIZE`,
`INGESTION_CHUNK_OVERLAP`, and `INGESTION_BATCH_SIZE` in your environment.
