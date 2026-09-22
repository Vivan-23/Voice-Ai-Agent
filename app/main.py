"""Voice AI Platform POC - Main Application Entrypoint."""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.conversation.conversation import Conversation
from app.agents.sales_agent import SalesAgent
from app.agents.support_agent import CustomerSupportAgent

app = FastAPI(title="Voice AI Platform POC", version="0.1.0")


class MessageRequest(BaseModel):
    department: str
    message: str
    history: Optional[List[Dict[str, str]]] = None


class MessageResponse(BaseModel):
    agent_name: str
    department: str
    response: str
    history: List[Dict[str, str]]
    llm_success: bool


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "voice_ai_platform_poc"}


@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    dept = request.department.upper()
    agent = SalesAgent() if dept == "SALES" else CustomerSupportAgent()
    conversation = Conversation(department=dept, active_agent=agent)
    if request.history:
        conversation.history = list(request.history)

    result = await conversation.process_message(request.message)
    return MessageResponse(
        agent_name=result.agent_name,
        department=result.department,
        response=result.text,
        history=conversation.history,
        llm_success=result.llm_result.success,
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
