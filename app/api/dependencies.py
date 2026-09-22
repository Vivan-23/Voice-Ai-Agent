"""FastAPI Dependency Injection providers."""

from functools import lru_cache
from config.settings import Settings, get_settings
from app.interfaces.stt import BaseSTTService
from app.interfaces.tts import BaseTTSService
from app.interfaces.knowledge import BaseKnowledgeService
from app.interfaces.crm import BaseCRMService
from app.services.audio.elevenlabs_stt import ElevenLabsSTTService
from app.services.audio.elevenlabs_tts import ElevenLabsTTSService
from app.services.knowledge.notebooklm_mcp import NotebookLMMCPKnowledgeService
from app.services.crm.local_crm import LocalCRMService
from app.services.post_call.processor import PostCallProcessor
from app.services.conversation import ConversationService
from app.agents.graph import create_call_agent_graph


@lru_cache()
def get_stt_service() -> BaseSTTService:
    """Return singleton STT provider (ElevenLabs)."""
    settings = get_settings()
    return ElevenLabsSTTService(api_key=settings.elevenlabs_api_key)


@lru_cache()
def get_tts_service() -> BaseTTSService:
    """Return singleton TTS provider (ElevenLabs)."""
    settings = get_settings()
    return ElevenLabsTTSService(
        api_key=settings.elevenlabs_api_key,
        default_voice_id=settings.elevenlabs_voice_id,
        model_id=settings.elevenlabs_model_id,
    )


from app.services.knowledge.hybrid import HybridCompanyKnowledgeService


@lru_cache()
def get_knowledge_service() -> BaseKnowledgeService:
    """Return singleton Knowledge provider (Fast Local Knowledge + NotebookLM MCP Fallback)."""
    settings = get_settings()
    mcp_fallback = None
    if settings.notebooklm_notebook_id or settings.notebooklm_notebook_url:
        mcp_fallback = NotebookLMMCPKnowledgeService(
            notebook_id=settings.notebooklm_notebook_id,
            notebook_url=settings.notebooklm_notebook_url,
        )
    return HybridCompanyKnowledgeService(fallback_service=mcp_fallback)


from langchain_core.language_models.chat_models import BaseChatModel
from app.core.llm import get_chat_model


@lru_cache()
def get_llm() -> BaseChatModel:
    """Return singleton configurable LLM provider."""
    return get_chat_model()


@lru_cache()
def get_crm_service() -> BaseCRMService:
    """Return singleton CRM provider."""
    return LocalCRMService()


@lru_cache()
def get_post_call_processor() -> PostCallProcessor:
    """Return post-call background processor."""
    return PostCallProcessor(crm_service=get_crm_service())


@lru_cache()
def get_agent_graph():
    """Return compiled LangGraph workflow."""
    knowledge_service = get_knowledge_service()
    llm = get_llm()
    return create_call_agent_graph(knowledge_service=knowledge_service, llm=llm)


@lru_cache()
def get_conversation_service() -> ConversationService:
    """Return singleton ConversationService instance."""
    knowledge_service = get_knowledge_service()
    llm = get_llm()
    return ConversationService(knowledge_service=knowledge_service, llm=llm)
