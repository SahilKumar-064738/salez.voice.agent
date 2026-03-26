import json
import base64
import time
import audioop
import asyncio
import httpx
import random

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from vosk import Model, KaldiRecognizer
import edge_tts

app = FastAPI()

import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")


FILLERS = [
    "Let me check that for you.",
    "Sure, one moment.",
    "Got it.",
]
# -------- LOAD DATA --------
with open("data.json", "r") as f:
    CLIENT_DATA = json.load(f)

# -------- GLOBALS --------
vosk_model = Model("vosk-model-small-en-us-0.15")

http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(3.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    http2=True
)

# -------- TIME --------
def now():
    return round(time.perf_counter() * 1000)

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

# -------- INTENT --------
KEYWORDS = {
    "gst": ["gst"],
    "income_tax": ["itr", "income tax", "tax"],
    "business_registration": ["company", "llp", "startup", "registration"]
}

def detect_intent(text):
    t = text.lower()
    for intent, words in KEYWORDS.items():
        if any(w in t for w in words):
            return intent
    return "general"

# -------- CONTEXT --------
def build_context(intent, query):
    try:
        chunks = []

        # services
        services = CLIENT_DATA.get("services", {})
        if intent in services:
            chunks.extend(services[intent])

        # pricing
        pricing = CLIENT_DATA.get("pricing", {})
        if intent in pricing:
            chunks.append(json.dumps(pricing[intent]))

        # documents
        docs = CLIENT_DATA.get("documents_required", {})
        if intent in docs:
            chunks.extend(docs[intent])

        # turnaround
        tat = CLIENT_DATA.get("turnaround_time", {})
        if intent in tat:
            chunks.append(f"Turnaround: {tat[intent]}")

        # FAQ matching
        query_words = set(query.lower().split())

        for faq in CLIENT_DATA.get("faqs", []):
                q_words = set(faq["question"].lower().split())
                if len(query_words & q_words) >= 2:
                    chunks.append(faq["answer"])
            

        return "\n".join(chunks[:6])

    except:
        return ""
# -------- GEMINI --------
async def ask_gemini(text, intent, history):
    t0 = now()

    context = build_context(intent, text)

    convo = "\n".join(history[-4:])  # last 4 turns only (fast)

    prompt = f"""
You are a CA IVR assistant.

STRICT RULES:
- Answer ONLY from the provided context
- If answer not in context → say EXACTLY: "Please contact us directly for details"
- DO NOT add extra knowledge
- Keep response under 2 sentences

CONTEXT:
{context}

CONVERSATION:
{convo}

USER:
{text}
"""

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        res = await http_client.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}]
        })

        if res.status_code == 200:
            data = res.json()
            if data.get("candidates"):
                print(f"⏱ GEMINI: {now()-t0} ms")
                return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        print("❌ Gemini:", e)

    return "We are Sharma CA Services based in Chandigarh."

# -------- ULTRA-FAST TTS --------
async def stream_tts(ws, text, sid, stop_flag):
    communicate = edge_tts.Communicate(
        text,
        voice="en-IN-NeerjaNeural",
        format="raw-16khz-16bit-mono-pcm"
    )

    async for chunk in communicate.stream():
        if stop_flag["stop"]:
            return

        if chunk["type"] == "audio":
            pcm = chunk["data"]

            # downsample to 8k
            pcm = audioop.ratecv(pcm, 2, 1, 16000, 8000, None)[0]
            mulaw = audioop.lin2ulaw(pcm, 2)

            for i in range(0, len(mulaw), 160):
                frame = mulaw[i:i+160]

                await ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": sid,
                    "media": {"payload": base64.b64encode(frame).decode()}
                }))



# -------- SESSION (FASTER TRIGGER) --------
class CallSession:
    def __init__(self):
        self.recognizer = KaldiRecognizer(vosk_model, 16000)
        self.recognizer.SetPartialWords(False)
        self.recognizer.SetWords(True)

        self.stream_sid = None
        self.last_speech = time.time()

        self.buffer = ""
        self.processing = False

        self.stop_flag = {"stop": False}
        self.tts_task = None

        self.start_time = None
        self.history = []
        self.lock = asyncio.Lock()

    async def process(self, ws, chunk):
        pcm = resample(mulaw_to_pcm(chunk))

        if self.start_time is None:
            self.start_time = now()

        is_final = self.recognizer.AcceptWaveform(pcm)

        partial = json.loads(self.recognizer.PartialResult())

        if partial.get("partial"):
            print("⚡", partial["partial"])

            # 🔥 INTERRUPT IMMEDIATELY
            if self.tts_task and not self.tts_task.done():
                self.stop_flag["stop"] = True
                self.stop_flag = {"stop": False}  # reset immediately

        if is_final:
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")

            if text:
                print("🗣️ FINAL:", text)
                print(f"⏱ STT: {now()-self.start_time} ms")

                self.buffer = text
                self.last_speech = time.time()
                self.start_time = None

        # 🔥 ULTRA FAST TRIGGER
        if self.buffer and not self.processing:
            if len(self.buffer.split()) >= 1 or (time.time() - self.last_speech > 0.1):
                self.processing = True
                asyncio.create_task(self.respond(ws, self.buffer))
                self.buffer = ""

    # -------- SMART RESPONSE ENGINE (NO GUESSING) --------
    def smart_reply(self, text):
        t = text.lower()

        # 🔥 RULE ENGINE (FAST + ACCURATE)
        if "who are you" in t:
            return "We are Sharma CA Services, a Chartered Accountant firm based in Chandigarh."

        if "where" in t:
            return "We are based in Chandigarh and serve clients across India."

        if "gst" in t:
            return "We provide GST registration, filing, and notice handling services."

        if "itr" in t or "tax" in t:
            return "We handle income tax filing, planning, and notices."

        if "company" in t or "registration" in t:
            return "We help with company, LLP, and startup registrations."

        if "price" in t:
            return "Our pricing is affordable and depends on the service required."

        return None  # fallback to Gemini
    async def respond(self, ws, text):
        async with self.lock:
            print("🚀", text)

            start = now()

            # 🔥 STEP 0: PLAY FILLER IMMEDIATELY
            filler = random.choice(FILLERS)

            filler_task = asyncio.create_task(
                stream_tts(ws, filler, self.stream_sid, {"stop": False})
            )

            # -------- STEP 1: PROCESS REAL ANSWER --------
            intent = detect_intent(text)

            instant = self.smart_reply(text)

            if instant:
                reply = instant
            else:
                reply = await ask_gemini(text, intent, self.history)

            # 🔥 STOP FILLER IF STILL PLAYING
            try:
                filler_task.cancel()
                await filler_task
            except:
                pass

            # -------- STEP 2: REAL RESPONSE --------
            self.stop_flag = {"stop": False}

            self.tts_task = asyncio.create_task(
                stream_tts(ws, reply, self.stream_sid, self.stop_flag)
            )

            # save memory
            self.history.append(f"User: {text}")
            self.history.append(f"Bot: {reply}")

            print(f"⏱ RESPONSE START: {now()-start} ms")

            self.processing = False
                
# -------- TWILIO --------
@app.post("/voice")
async def voice():
    return Response(content="""
    <Response>
        <Say>Welcome to Sharma CA Services.</Say>
        <Connect>
            <Stream url="wss://{NGROK_DOMAIN}/audio" />
        </Connect>
    </Response>
    """, media_type="application/xml")

# -------- WS --------
@app.websocket("/audio")
async def audio(ws: WebSocket):
    await ws.accept()
    print("🔥 Call connected")

    session = CallSession()

    while True:
        msg = await ws.receive_text()
        data = json.loads(msg)

        if data["event"] == "start":
            session.stream_sid = data["start"]["streamSid"]

        elif data["event"] == "media":
            chunk = base64.b64decode(data["media"]["payload"])
            await session.process(ws, chunk)

        elif data["event"] == "stop":
            print("🛑 Call ended")
            session.history.clear()
            break