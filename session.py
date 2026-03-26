import time
import json
import asyncio
import random

from vosk import KaldiRecognizer

from audio import stream_tts, mulaw_to_pcm, resample
from ai import detect_intent, ask_gemini_streaming
FILLERS = [
    "Let me check that for you.",
    "Sure, one moment.",
    "Got it.",
]

def now():
    return round(time.perf_counter() * 1000)


class CallSession:
    def __init__(self, client_data, vosk_model):
        self.client_data = client_data

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
        self.pending_text = None  # track latest user text

    def cleanup(self):
        self.stop_flag["stop"] = True
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
        print(f"🧹 Session cleaned up | turns={len(self.history)//2}")
        
    async def _clear_twilio(self, ws):
        try:
            await ws.send_text(json.dumps({
                "event": "clear",
                "streamSid": self.stream_sid
            }))
        except Exception:
            pass
    # -------- AUDIO PROCESS --------
    async def process(self, ws, chunk):
        loop = asyncio.get_event_loop()
        # Offload CPU work to thread pool
        pcm = await loop.run_in_executor(
            None,
            lambda: resample(mulaw_to_pcm(chunk))
        )

        if self.start_time is None:
            self.start_time = now()

        is_final = self.recognizer.AcceptWaveform(pcm)
        partial = json.loads(self.recognizer.PartialResult())

        # 🔥 PARTIAL (for interruption)
        if partial.get("partial"):
            print("⚡", partial["partial"])

            if self.tts_task and not self.tts_task.done():
                self.stop_flag["stop"] = True
                self.stop_flag = {"stop": False}
                asyncio.create_task(self._clear_twilio(ws))  # ← ADD THIS


        # 🔥 FINAL TEXT
        if is_final:
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")

            if text:
                print("🗣️ FINAL:", text)
                print(f"⏱ STT: {now()-self.start_time} ms")

                self.buffer = text
                self.last_speech = time.time()
                self.start_time = None

        # 🔥 SMART TRIGGER (NO NOISE)
        if self.buffer and not self.processing:
            if len(self.buffer.split()) >= 2 or (time.time() - self.last_speech > 0.3):
                self.processing = True
                asyncio.create_task(self.respond(ws, self.buffer))
                self.buffer = ""

    # -------- RULE ENGINE --------
    def smart_reply(self, text: str) -> str | None:
        t = text.lower()
        identity = self.client_data.get("business_identity", {})
        guidelines = self.client_data.get("ai_assistant_guidelines", {})

        # Who are you?
        if "who are you" in t or "what is your name" in t:
            name = identity.get("trade_name") or identity.get("legal_name", "")
            return f"We are {name}." if name else None

        # Location
        if "where" in t or "location" in t or "address" in t:
            office = identity.get("offices", {}).get("head", {})
            city = office.get("city", "")
            return f"We are based in {city} and serve clients across India." if city else None

        # Service keyword hits from tenant data
        services = self.client_data.get("services", {})
        for svc_key, svc_items in services.items():
            # Use first service item as a short descriptor
            if svc_key.replace("_", " ") in t and svc_items:
                return svc_items[0] if isinstance(svc_items[0], str) else None

        return None  # fall through to Gemini
    # -------- RESPONSE ENGINE --------
    async def respond(self, ws, text):
        async with self.lock:
            try:
                start = now()
                spoken_via_streaming = False  # ← track this

                if self.tts_task and not self.tts_task.done():
                    self.stop_flag["stop"] = True
                    try:
                        await asyncio.wait_for(self.tts_task, timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

                self.stop_flag = {"stop": False}

                intent = detect_intent(text)
                instant = self.smart_reply(text)

                filler_task = None
                reply = None

                if instant:
                    reply = instant
                else:
                    filler_task = asyncio.create_task(
                        stream_tts(ws, random.choice(FILLERS), self.stream_sid, self.stop_flag)
                    )
                    try:
                        sentences = []
                        async for sentence in ask_gemini_streaming(text, intent, self.history, self.client_data):
                            sentences.append(sentence)

                            if len(sentences) == 1:
                                # Cancel filler BEFORE first sentence TTS
                                if filler_task and not filler_task.done():
                                    filler_task.cancel()
                                    try:
                                        await filler_task
                                    except asyncio.CancelledError:
                                        pass
                                    filler_task = None

                                await self._clear_twilio(ws)
                                self.stop_flag = {"stop": False}
                                self.tts_task = asyncio.create_task(
                                    stream_tts(ws, sentence, self.stream_sid, self.stop_flag)
                                )
                                await self.tts_task
                                spoken_via_streaming = True  # ← mark as spoken

                        reply = " ".join(sentences).strip() if sentences else None
                        if not reply:
                            raise ValueError("empty Gemini response")

                    except Exception as e:
                        print(f"❌ Gemini error: {e}")
                        identity = self.client_data.get("business_identity", {})
                        name = identity.get("trade_name") or identity.get("legal_name", "our team")
                        reply = f"I don't have that detail right now. Please contact {name} directly."
                    finally:
                        if filler_task and not filler_task.done():
                            filler_task.cancel()
                            try:
                                await filler_task
                            except asyncio.CancelledError:
                                pass

                # Only speak here if NOT already spoken via streaming
                if reply and not spoken_via_streaming:
                    await self._clear_twilio(ws)
                    self.stop_flag = {"stop": False}
                    self.tts_task = asyncio.create_task(
                        stream_tts(ws, reply, self.stream_sid, self.stop_flag)
                    )

                if reply:
                    self.history.append(f"User: {text}")
                    self.history.append(f"Bot: {reply}")
                    if len(self.history) > 10:
                        self.history = self.history[-6:]

                print(f"⏱ RESPONSE START: {now()-start} ms")

            except Exception as e:
                print(f"❌ respond() crash: {e}")
            finally:
                self.processing = False