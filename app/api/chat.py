from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, ChatSource
from app.config import ChatSettings
from app.services import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    try:
        settings = ChatSettings.from_env()
        answer, sources = answer_question(question=question, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Chat completion failed.") from exc

    return ChatResponse(answer=answer, sources=[ChatSource(**source) for source in sources])
