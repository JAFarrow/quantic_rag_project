from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.routes import router as api_router
from app.config import APP_VERSION
from app.web import router as web_router


def create_application() -> FastAPI:
    load_dotenv()
    app = FastAPI(title="Quantic RAG Project API", version=APP_VERSION)
    app.include_router(web_router)
    app.include_router(api_router)
    return app


app = create_application()
