"""WebSocket Real-Time Voice Streaming Endpoint.

Handles full-duplex bi-directional audio:
Customer Audio Input (Bytes) -> STT -> Agent Graph -> Knowledge (MCP) -> TTS -> Customer Audio Output (Bytes)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from langchain_core.messages import HumanMessage
from app.api.dependencies import get_stt_service, get_tts_service, get_agent_graph
from app.interfaces.stt import BaseSTTService
from app.interfaces.tts import BaseTTSService
from app.schemas.agent import AgentRole
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio Stream"])


@router.websocket("/stream/{call_id}")
async def audio_stream_websocket(
    websocket: WebSocket,
    call_id: str,
    stt_service: BaseSTTService = Depends(get_stt_service),
    tts_service: BaseTTSService = Depends(get_tts_service),
    agent_graph = Depends(get_agent_graph),
):
    """Full-duplex WebSocket stream for conversational voice interaction."""
    await websocket.accept()
    logger.info(f"WebSocket voice connection opened for call_id={call_id}")

    try:
        while True:
            # Receive incoming audio data or message frame from client/telephony bridge
            data = await websocket.receive_bytes()
            if not data:
                continue

            # 1. Speech-to-Text via ElevenLabs STT
            transcript_segment = await stt_service.transcribe_file(data)
            logger.info(f"Transcribed customer turn: '{transcript_segment.text}'")

            # 2. Feed customer message to LangGraph Multi-Agent Orchestrator
            initial_state = {
                "messages": [HumanMessage(content=transcript_segment.text)],
                "call_id": call_id,
                "customer_phone": "incoming_caller",
                "active_agent": AgentRole.CUSTOMER_SERVICE,
                "previous_agent": None,
                "retrieved_knowledge": [],
                "knowledge_grounded": True,
                "needs_manager": False,
                "manager_trigger_reason": None,
                "manager_interventions_count": 0,
                "manager_guidance": None,
                "human_escalation_required": False,
                "customer_frustrated": False,
                "frustration_score": 0.0,
                "turn_uncertainty": False,
                "consecutive_failures": 0,
            }

            final_state = await agent_graph.ainvoke(initial_state)
            last_agent_message = final_state["messages"][-1].content

            # 3. Text-to-Speech via ElevenLabs TTS
            synthesized_audio = await tts_service.synthesize_text(str(last_agent_message))

            # 4. Stream synthesized audio back to caller
            await websocket.send_bytes(synthesized_audio.data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for call_id={call_id}")
    except Exception as e:
        logger.error(f"WebSocket streaming error for call_id={call_id}: {e}")
        await websocket.close()
