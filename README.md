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

