"""Setup script for ElevenLabs Speech Engine resource.

Creates or verifies the Speech Engine resource using the official ElevenLabs SDK,
pointing to our publicly reachable WebSocket endpoint (/ws/voice).

Usage:
    python scripts/setup_elevenlabs.py
    python scripts/setup_elevenlabs.py --ws-url wss://xyz.ngrok-free.app/ws/voice
"""

import os
import sys
import argparse
import urllib.request
import json
from typing import Optional
from dotenv import load_dotenv, find_dotenv

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv(find_dotenv())

from config.settings import get_settings
from elevenlabs.client import ElevenLabs
from elevenlabs.types.speech_engine_config import SpeechEngineConfig


def auto_detect_ngrok_url() -> Optional[str]:
    """Attempt to detect active ngrok public HTTPS tunnel from local ngrok client API."""
    try:
        req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tunnels = data.get("tunnels", [])
            for tunnel in tunnels:
                public_url = tunnel.get("public_url", "")
                if public_url.startswith("https://"):
                    return public_url
    except Exception:
        pass
    return None


def normalize_ws_url(raw_url: str) -> str:
    """Normalize HTTP/HTTPS or WS/WSS URL to wss://.../ws/voice endpoint."""
    url = raw_url.strip()
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    elif not (url.startswith("wss://") or url.startswith("ws://")):
        url = "wss://" + url

    url = url.rstrip("/")
    if not url.endswith("/ws/voice"):
        url = f"{url}/ws/voice"
    return url


def main():
    parser = argparse.ArgumentParser(description="Configure ElevenLabs Speech Engine for Coway AI")
    parser.add_argument(
        "--ws-url",
        dest="ws_url",
        help="Public WebSocket URL pointing to /ws/voice (e.g. wss://xyz.ngrok-free.app/ws/voice)",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")

    print("=" * 65)
    print("  Coway Voice AI Platform — ElevenLabs Speech Engine Setup")
    print("=" * 65)

    if not api_key or api_key.strip() in ("", "your_elevenlabs_api_key_here"):
        print("\n[ERROR] ELEVENLABS_API_KEY is not configured in .env file.")
        print("Please configure ELEVENLABS_API_KEY in .env before running this setup.\n")
        sys.exit(1)

    # 1. Determine Public WebSocket URL
    ws_url = args.ws_url or settings.elevenlabs_public_ws_url or os.getenv("ELEVENLABS_PUBLIC_WS_URL")
    if not ws_url:
        ngrok_detected = auto_detect_ngrok_url()
        if ngrok_detected:
            ws_url = normalize_ws_url(ngrok_detected)
            print(f"\n[AUTO-DETECT] Found active ngrok tunnel: {ngrok_detected}")
            print(f"[AUTO-DETECT] Setting WebSocket target: {ws_url}")
        else:
            print("\n[ERROR] Public WebSocket URL is required so ElevenLabs can connect to /ws/voice.")
            print("Please provide it via:")
            print("  python scripts/setup_elevenlabs.py --ws-url wss://YOUR_DOMAIN.ngrok-free.app/ws/voice")
            print("or set ELEVENLABS_PUBLIC_WS_URL in your .env file.\n")
            print("To start ngrok locally, run:")
            print("  ngrok http 8000\n")
            sys.exit(1)
    else:
        ws_url = normalize_ws_url(ws_url)

    print(f"\nTarget WebSocket URL: {ws_url}")

    # 2. Check if Speech Engine already exists
    existing_id = settings.elevenlabs_speech_engine_id or os.getenv("ELEVENLABS_SPEECH_ENGINE_ID")
    client = ElevenLabs(api_key=api_key)

    if existing_id and existing_id.strip() not in ("", "your_speech_engine_id_here"):
        print(f"\n[CONFIGURED] Existing Speech Engine ID found: {existing_id}")
        try:
            print("Verifying Speech Engine with ElevenLabs...")
            engine_resource = client.speech_engine.get(existing_id)
            print(f"[SUCCESS] Speech Engine {existing_id} is valid and verified.")
            print(f"\nTo use this engine, ensure .env contains:")
            print(f"  ELEVENLABS_SPEECH_ENGINE_ID={existing_id}")
            print(f"  ELEVENLABS_PUBLIC_WS_URL={ws_url}")
            print("=" * 65 + "\n")
            return
        except Exception as e:
            err_str = str(e)
            if "convai_read" in err_str or "missing_permissions" in err_str:
                print(f"\n[NOTICE] Your API key is valid but lacks 'convai_read' permission to query engine status.")
                print(f"[NOTICE] Using configured ELEVENLABS_SPEECH_ENGINE_ID={existing_id} as specified.")
                print("=" * 65 + "\n")
                return
            else:
                print(f"[WARNING] Could not retrieve existing engine ({e}). Attempting to create new engine...")

    # 3. Create Speech Engine Resource
    print("\nCreating ElevenLabs Speech Engine resource...")
    try:
        from elevenlabs.types.speech_engine_conversation_initiation_client_data_config import SpeechEngineConversationInitiationClientDataConfig
        config = SpeechEngineConfig(ws_url=ws_url)
        resource = client.speech_engine.create(
            name="Coway AI Customer Service & Sales POC",
            speech_engine=config,
            overrides=SpeechEngineConversationInitiationClientDataConfig(first_message=True),
        )
        engine_id = (
            getattr(resource, "engine_id", None)
            or getattr(resource, "speech_engine_id", None)
            or getattr(resource, "id", None)
        )
        if not engine_id and hasattr(resource, "config") and resource.config:
            engine_id = getattr(resource.config, "speech_engine_id", None) or getattr(resource.config, "id", None)

        print(f"\n[SUCCESS] Speech Engine created successfully!")
        print(f"  Speech Engine ID: {engine_id}")
        print("\nPlease add the following lines to your .env file:")
        print("-" * 55)
        print(f"ELEVENLABS_SPEECH_ENGINE_ID={engine_id}")
        print(f"ELEVENLABS_PUBLIC_WS_URL={ws_url}")
        print("-" * 55)
        print("=" * 65 + "\n")

    except Exception as e:
        err_str = str(e)
        print(f"\n[ERROR] Failed to create Speech Engine: {err_str}")
        if "missing_permissions" in err_str or "convai_write" in err_str:
            print("\nPERMISSION NOTICE:")
            print("The ELEVENLABS_API_KEY in your .env requires Conversational AI scopes.")
            print("Please ensure your API key has 'convai_write' & 'convai_read' permissions enabled")
            print("in the ElevenLabs developer console (https://elevenlabs.io/app/settings/api-keys).")
            print("Alternatively, if you created an agent or speech engine manually in the dashboard,")
            print("simply copy its ID and set it in your .env:")
            print("  ELEVENLABS_SPEECH_ENGINE_ID=seng_YOUR_ID_HERE\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
