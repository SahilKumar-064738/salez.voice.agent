import os
import httpx
import time
import json


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(3.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    http2=True
)

def now():
    return round(time.perf_counter() * 1000)



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
def build_context(client_data,intent, query):
    try:
        chunks = []

        # services
        services = client_data.get("services", {})
        if intent in services:
            chunks.extend(services[intent])

        # pricing
        pricing = client_data.get("pricing", {})
        if intent in pricing:
            chunks.append(json.dumps(pricing[intent]))

        # documents
        docs = client_data.get("documents_required", {})
        if intent in docs:
            chunks.extend(docs[intent])

        # turnaround
        tat = client_data.get("turnaround_time", {})
        if intent in tat:
            chunks.append(f"Turnaround: {tat[intent]}")

        # FAQ matching
        query_words = set(query.lower().split())

        # Replace FAQ section:
        for faq in client_data.get("_faq_index", []):
            if len(query_words & faq["q_words"]) >= 2:
                chunks.append(faq["answer"])
            

        return "\n".join(chunks[:6])

    except:
        return ""
# -------- GEMINI --------
async def ask_gemini(text, intent, history,client_data):
    t0 = now()

    context = build_context(client_data, intent, text)

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

    return None

async def ask_gemini_streaming(text, intent, history, client_data):
    """Yields complete sentences as they arrive from Gemini."""
    context = build_context(client_data, intent, text)
    convo = "\n".join(history[-4:])
    prompt = f"""You are an IVR assistant for {client_data.get('business_identity', {}).get('trade_name', 'this business')}.
RULES: Answer ONLY from context. Max 2 sentences. If unknown say: "Please contact us directly."
CONTEXT:\n{context}\nCONVERSATION:\n{convo}\nUSER:\n{text}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    buffer = ""
    async with http_client.stream("POST", url, json={"contents": [{"parts": [{"text": prompt}]}]}) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                token = data["candidates"][0]["content"]["parts"][0]["text"]
                buffer += token
                # Yield on sentence boundary
                for delim in [".", "!", "?"]:
                    idx = buffer.find(delim)
                    if idx != -1:
                        sentence = buffer[:idx+1].strip()
                        buffer = buffer[idx+1:].strip()
                        if sentence:
                            yield sentence
            except Exception:
                continue
    if buffer.strip():
        yield buffer.strip()