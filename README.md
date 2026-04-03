# quantic_rag_project

LLM-powered chatbot foundation with room for a policy-aware RAG stack.

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
