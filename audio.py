import base64
import audioop
import edge_tts
import asyncio
import json

# -------- AUDIO --------
def mulaw_to_pcm(mu):
    return audioop.ulaw2lin(mu, 2)

def pcm_to_mulaw(pcm):
    mulaw = audioop.lin2ulaw(pcm.tobytes(), 2)
    if len(mulaw) % 160 != 0:
        mulaw += b'\xff' * (160 - len(mulaw) % 160)
    return mulaw

def resample(pcm):
    return audioop.ratecv(pcm, 2, 1, 8000, 16000, None)[0]


# -------- ULTRA-FAST TTS --------
async def stream_tts(ws, text, sid, stop_flag):
    communicate = edge_tts.Communicate(
        text, voice="en-IN-NeerjaNeural",
        format="raw-16khz-16bit-mono-pcm"
    )
    frame_buffer = b""
    BATCH_SIZE = 640  # 4 × 160 bytes = 4 frames

    async for chunk in communicate.stream():
        if stop_flag["stop"]:
            return
        if chunk["type"] != "audio":
            continue

        pcm = audioop.ratecv(chunk["data"], 2, 1, 16000, 8000, None)[0]
        frame_buffer += audioop.lin2ulaw(pcm, 2)

        while len(frame_buffer) >= BATCH_SIZE:
            batch = frame_buffer[:BATCH_SIZE]
            frame_buffer = frame_buffer[BATCH_SIZE:]
            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": sid,
                "media": {"payload": base64.b64encode(batch).decode()}
            }))
            await asyncio.sleep(0)

    # Flush remainder
    if frame_buffer and not stop_flag["stop"]:
        await ws.send_text(json.dumps({
            "event": "media", "streamSid": sid,
            "media": {"payload": base64.b64encode(frame_buffer).decode()}
        }))
