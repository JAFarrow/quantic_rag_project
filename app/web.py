from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter()

_INDEX_PATH = Path(__file__).with_name("static") / "index.html"


@router.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(_INDEX_PATH.read_text(encoding="utf-8"))
