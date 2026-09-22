"""Custom domain exceptions for Voice AI Platform."""


class VoiceAIException(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class STTException(VoiceAIException):
    """Raised when speech-to-text transcription fails."""
    pass


class TTSException(VoiceAIException):
    """Raised when text-to-speech synthesis fails."""
    pass


class KnowledgeRetrievalException(VoiceAIException):
    """Raised when knowledge retrieval via MCP or local store fails."""
    pass


class AgentOrchestrationException(VoiceAIException):
    """Raised when agent execution or state graph routing fails."""
    pass


class CRMIntegrationException(VoiceAIException):
    """Raised when syncing customer data with CRM fails."""
    pass
