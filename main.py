from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
import json
import base64
import os
import asyncio

from vosk import Model
from tenants.loader import load_tenants, get_tenant
from audio import stream_tts
from session import CallSession

app = FastAPI()

NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")

# load vosk once
vosk_model = Model("vosk-model-small-en-us-0.15")

# load tenants
load_tenants()
from tenants.loader import reload_tenants

@app.post("/admin/reload-tenants")
async def reload(secret: str):
    if secret != os.getenv("ADMIN_SECRET"):
        return {"error": "unauthorized"}
    reload_tenants()
    return {"status": "reloaded"}

@app.post("/voice/{tenant}")
async def voice(tenant: str):
    return Response(content=f"""
    <Response>
        <Connect>
            <Stream url="wss://{NGROK_DOMAIN}/audio/{tenant}" />
        </Connect>
    </Response>
    """, media_type="application/xml")


@app.websocket("/audio/{tenant}")
async def audio(ws: WebSocket, tenant: str):
    await ws.accept()
    client_data = get_tenant(tenant)
    if not client_data:
        await ws.close(code=1008)
        return

    session = CallSession(client_data, vosk_model)
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data["event"] == "start":
                session.stream_sid = data["start"]["streamSid"]
                greeting = client_data.get("ai_assistant_guidelines", {}).get(
                    "greeting", "Welcome. How can I help you today?"
                )
                asyncio.create_task(
                    stream_tts(ws, greeting, session.stream_sid, {"stop": False})
                )
            elif data["event"] == "media":
                chunk = base64.b64decode(data["media"]["payload"])
                await session.process(ws, chunk)
            elif data["event"] == "stop":
                break

    except Exception as e:
        print(f"❌ WebSocket error [{tenant}]: {e}")
    finally:
        session.cleanup()          # always run, even on crash