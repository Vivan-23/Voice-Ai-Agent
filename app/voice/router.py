"""FastAPI router for ElevenLabs Speech Engine voice integration.

Includes:
- WebSocket endpoint (/ws/voice) for real-time speech engine streaming.
- REST endpoints (/api/voice/session, /api/voice/token, /api/voice/session/bind)
  for client credential generation, department selection, and session isolation.
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, status
from pydantic import BaseModel
from elevenlabs.client import ElevenLabs
from elevenlabs.speech_engine import SpeechEngineResource, verify_speech_engine_jwt

from config.settings import get_settings
from app.voice.session_manager import (
    get_voice_session_manager,
    VoiceSession,
    VoiceTurnTiming,
)

logger = logging.getLogger("app.voice.router")

router = APIRouter(tags=["Voice"])


class VoiceSessionRequest(BaseModel):
    department: str = "SALES"


class VoiceSessionResponse(BaseModel):
    session_id: str
    department: str
    agent_name: str
    first_message: str
    signed_url: str


class VoiceBindRequest(BaseModel):
    conversation_id: str
    session_id: Optional[str] = None
    department: Optional[str] = None


@router.post("/api/voice/session", response_model=VoiceSessionResponse)
async def create_voice_session(request: VoiceSessionRequest):
    """Create a reserved voice session with selected department and fetch ElevenLabs signed URL."""
    return await _initiate_voice_session(request.department)


@router.get("/api/voice/token", response_model=VoiceSessionResponse)
async def get_voice_token(department: str = Query("SALES", description="Department: SALES or CUSTOMER_SUPPORT")):
    """GET endpoint to fetch voice session signed URL for the browser client."""
    return await _initiate_voice_session(department)


async def _initiate_voice_session(department: str) -> VoiceSessionResponse:
    """Shared helper to validate credentials and generate a signed URL for Speech Engine."""
    settings = get_settings()

    if not settings.elevenlabs_api_key or settings.elevenlabs_api_key.strip() in ("", "your_elevenlabs_api_key_here"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ELEVENLABS_API_KEY is not configured in .env file.",
        )

    if not settings.elevenlabs_speech_engine_id or settings.elevenlabs_speech_engine_id.strip() in ("", "your_speech_engine_id_here"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "ELEVENLABS_SPEECH_ENGINE_ID is not configured in .env file. "
                "Please run: python scripts/setup_elevenlabs.py to create or configure your Speech Engine."
            ),
        )

    session_mgr = get_voice_session_manager()
    session = session_mgr.create_pending_session(department)

    try:
        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        # Request short-lived signed URL for client to connect to Speech Engine
        signed_url_resp = client.conversational_ai.conversations.get_signed_url(
            agent_id=settings.elevenlabs_speech_engine_id
        )
        signed_url = signed_url_resp.signed_url
    except Exception as e:
        logger.error(f"Failed to obtain signed URL from ElevenLabs: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ElevenLabs API error when generating signed URL: {str(e)}",
        )

    return VoiceSessionResponse(
        session_id=session.session_id,
        department=session.department,
        agent_name=session.agent_name,
        first_message=session.greeting,
        signed_url=signed_url,
    )


@router.post("/api/voice/session/bind")
async def bind_voice_conversation(request: VoiceBindRequest):
    """Bind client conversation_id to the reserved VoiceSession."""
    session_mgr = get_voice_session_manager()
    session = session_mgr.bind_conversation(
        conversation_id=request.conversation_id,
        session_id=request.session_id,
        department=request.department,
    )
    return {
        "status": "ok",
        "conversation_id": request.conversation_id,
        "session_id": session.session_id,
        "agent_name": session.agent_name,
        "department": session.department,
    }


@router.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket):
    """Dedicated WebSocket endpoint for ElevenLabs Speech Engine server-side integration."""
    settings = get_settings()
    headers = dict(websocket.headers)

    # 1. Official ElevenLabs Speech Engine JWT Authorization Verification
    auth_header = headers.get("x-elevenlabs-speech-engine-authorization")
    if settings.elevenlabs_api_key and auth_header:
        try:
            verify_speech_engine_jwt(auth_header, settings.elevenlabs_api_key)
            logger.info("ElevenLabs Speech Engine JWT verified successfully.")
        except Exception as e:
            logger.warning(f"Speech Engine authorization failed: {e}")
            await websocket.close(code=1008)
            return

    # 2. Accept WebSocket connection
    await websocket.accept()

    # 3. Create Speech Engine Resource and Session
    engine_id = settings.elevenlabs_speech_engine_id or "seng_default"
    engine = SpeechEngineResource(engine_id=engine_id)
    speech_session = engine.create_session(websocket, debug=settings.app_debug)

    session_mgr = get_voice_session_manager()
    active_voice_session: Optional[VoiceSession] = None
    current_conv_id: Optional[str] = None

    # 4. Wire Speech Engine lifecycle events
    @speech_session.on("init")
    async def handle_init(conversation_id: str):
        nonlocal active_voice_session, current_conv_id
        current_conv_id = conversation_id
        active_voice_session = session_mgr.get_session(conversation_id)
        if not active_voice_session:
            active_voice_session = session_mgr.bind_conversation(conversation_id)
        logger.info(
            f"Voice Session Init: conv_id={conversation_id}, "
            f"agent={active_voice_session.agent_name}, dept={active_voice_session.department}"
        )

    @speech_session.on("user_transcript")
    async def handle_user_transcript(transcript: List[Any]):
        nonlocal active_voice_session, current_conv_id
        t_stt_received = time.perf_counter()

        # Extract latest user message from transcript
        customer_utterance = ""
        for msg in reversed(transcript):
            role = getattr(msg, "role", "")
            if role == "user":
                customer_utterance = getattr(msg, "content", "").strip()
                break

        if not customer_utterance:
            return

        if not active_voice_session:
            if current_conv_id:
                active_voice_session = session_mgr.get_session(current_conv_id)
            if not active_voice_session:
                active_voice_session = session_mgr.create_pending_session("SALES")

        t_turn_start = time.perf_counter()
        stt_latency = t_turn_start - t_stt_received

        try:
            # Process turn through existing Conversation logic (Groq + Local Knowledge)
            response = await active_voice_session.conversation.process_message(customer_utterance)

            t_proc_done = time.perf_counter()
            first_text_latency = t_proc_done - t_turn_start

            # Send response to ElevenLabs Speech Engine for TTS synthesis
            # SpeechEngineSession handles task cancellation automatically on user interruption
            await speech_session.send_response(response.text)

            t_turn_end = time.perf_counter()
            total_turn_sec = t_turn_end - t_turn_start

            # Collect timing telemetry
            timings = response.timings
            voice_timing = VoiceTurnTiming(
                transcript=customer_utterance,
                turn_total_seconds=total_turn_sec,
                stt_seconds=stt_latency if stt_latency > 0 else None,
                llm_initial_seconds=timings.llm_initial_seconds if timings else 0.0,
                knowledge_seconds=timings.knowledge_seconds if timings else None,
                llm_final_seconds=timings.llm_final_seconds if timings else None,
                time_to_first_text_seconds=first_text_latency,
                llm_provider=timings.llm_provider if timings else "GROQ",
                knowledge_provider="LOCAL",
                notebooklm_called=False,
                manager_invoked=response.manager_invoked,
                manager_action=response.manager_action,
                assigned_agent=response.assigned_agent,
                agent_name=active_voice_session.agent_name,
                department=active_voice_session.department,
            )
            active_voice_session.record_turn(voice_timing)

            # Output voice telemetry per spec
            if settings.app_debug:
                print(voice_timing.format_debug_block())

        except asyncio.CancelledError:
            # Barge-in / interruption event: ElevenLabs cancelled previous response task
            logger.info("Voice turn cancelled by Speech Engine due to user barge-in / interruption.")
            raise
        except Exception as e:
            logger.error(f"Error during voice turn processing: {e}", exc_info=True)
            fallback_text = "I'm sorry, I encountered a brief glitch. Could you repeat that?"
            await speech_session.send_response(fallback_text)

    @speech_session.on("close")
    async def handle_close():
        nonlocal current_conv_id
        if current_conv_id:
            session_mgr.remove_session(current_conv_id)
            logger.info(f"Voice Session closed: conv_id={current_conv_id}")

    # 5. Run the Speech Engine session loop
    try:
        await speech_session.run()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
        if current_conv_id:
            session_mgr.remove_session(current_conv_id)
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
        if current_conv_id:
            session_mgr.remove_session(current_conv_id)
