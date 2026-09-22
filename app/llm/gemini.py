"""LLM Client for Voice AI Platform POC (Supports Groq & Gemini).

Provides a direct wrapper around Groq (Fast Inference) and Google Gemini models via LangChain.
Normalizes all responses into clean spoken text without metadata leaks.
"""

import os
import logging
from typing import List, Optional, Any
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel

load_dotenv(find_dotenv())

# Suppress Google GenAI internal AFC advisory notices
logging.getLogger("google.genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


def normalize_llm_text(content: Any) -> str:
    """Extract clean string text from various LLM response formats (str, list of dicts, AIMessage).
    
    Prevents leaking internal dictionaries, tool signatures, extras, or Python list representations,
    and strips emojis that cause console encoding crashes and poor phone TTS output.
    """
    if not content:
        return ""
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    text_parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                text_parts.append(item.text)
        text = "".join(text_parts).strip()
    else:
        text = str(content).strip()

    # Replace common unicode typographic characters with ASCII equivalents
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
        "₹": "INR ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Clean to ASCII-safe representation for phone TTS and standard terminals
    return text.encode("ascii", errors="ignore").decode("ascii").strip()


class LLMCallResult(BaseModel):
    """Result of an LLM generation call with execution metadata."""
    content: str
    provider: str = "GROQ"
    model: str
    success: bool = True
    error_message: Optional[str] = None


class GeminiClient:
    """Conversational LLM client supporting Groq (primary high-speed) and Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "groq").upper()
        self.temperature = temperature
        self._llm = None
        self._is_configured = False

        if self.provider == "GROQ":
            if api_key is not None:
                self.api_key = api_key
            else:
                self.api_key = os.getenv("GROQ_API_KEY")
            
            self.model_name = model_name or os.getenv("LLM_MODEL") or "openai/gpt-oss-20b"

            if self.api_key and self.api_key.strip() not in ("", "your_groq_api_key_here"):
                try:
                    from langchain_groq import ChatGroq
                    self._llm = ChatGroq(
                        model=self.model_name,
                        api_key=self.api_key,
                        temperature=self.temperature,
                    )
                    self._is_configured = True
                except Exception:
                    self._is_configured = False
        else:
            # Fallback / Gemini Provider
            self.provider = "GEMINI"
            if api_key is not None:
                self.api_key = api_key
            else:
                self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

            self.model_name = model_name or os.getenv("LLM_MODEL") or "gemini-3.6-flash"

            if self.api_key and self.api_key.strip() not in ("", "your_gemini_api_key_here"):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self._llm = ChatGoogleGenerativeAI(
                        model=self.model_name,
                        google_api_key=self.api_key,
                        temperature=self.temperature,
                    )
                    self._is_configured = True
                except Exception:
                    self._is_configured = False

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    async def generate(self, messages: List[BaseMessage]) -> LLMCallResult:
        """Execute a conversational completion with the configured LLM model."""
        if not self._is_configured or self._llm is None:
            err = (
                f"{self.provider}_API_KEY is not configured. "
                f"Please set {self.provider}_API_KEY in your .env file."
            )
            return LLMCallResult(
                content="",
                provider=self.provider,
                model=self.model_name,
                success=False,
                error_message=err,
            )

        try:
            response = await self._llm.ainvoke(messages)
            clean_text = normalize_llm_text(response.content)
            return LLMCallResult(
                content=clean_text,
                provider=self.provider,
                model=self.model_name,
                success=True,
            )
        except Exception as e:
            return LLMCallResult(
                content="",
                provider=self.provider,
                model=self.model_name,
                success=False,
                error_message=f"{self.provider} API error: {str(e)}",
            )

    def generate_sync(self, messages: List[BaseMessage]) -> LLMCallResult:
        """Synchronous generation wrapper."""
        if not self._is_configured or self._llm is None:
            err = (
                f"{self.provider}_API_KEY is not configured. "
                f"Please set {self.provider}_API_KEY in your .env file."
            )
            return LLMCallResult(
                content="",
                provider=self.provider,
                model=self.model_name,
                success=False,
                error_message=err,
            )

        try:
            response = self._llm.invoke(messages)
            clean_text = normalize_llm_text(response.content)
            return LLMCallResult(
                content=clean_text,
                provider=self.provider,
                model=self.model_name,
                success=True,
            )
        except Exception as e:
            return LLMCallResult(
                content="",
                provider=self.provider,
                model=self.model_name,
                success=False,
                error_message=f"{self.provider} API error: {str(e)}",
            )
