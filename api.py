"""Local API server wrapping chatbot.answer(), called by the Streamlit app over the network.

Run:
    pip install fastapi uvicorn python-multipart
    python api.py            # loads the model first (a few minutes), then serves on :8000
Then, in a second terminal:
    ngrok http 8000          # copy the https URL into B_API_URL in the app's .env
"""
import base64
import binascii
import threading
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()          # read GROQ_API_KEY / LLM_BACKEND from .env before chatbot is imported
except ImportError:
    pass

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chatbot import LLMUnavailable, answer

app = FastAPI(title="MediCheck Sverige API")
_gpu_lock = threading.Lock()   # one generation at a time: the model/GPU is shared


class AnswerRequest(BaseModel):
    text: str = ""
    image_b64: Optional[str] = None   # base64 image bytes (a data: URL prefix is accepted), or null
    state: Optional[dict] = None      # the app sends null on the first turn


class AnswerResponse(BaseModel):
    status: str
    answer: str
    drugs: list
    severity: Optional[str] = None
    citations: list
    citation_details: Optional[list] = None       # [{label, url}] readable sources for display
    state: dict
    safety_flag: Optional[str] = None             # 'crisis' | 'dose' | None
    gap: Optional[bool] = None                    # True: fixed 'no FASS data' reply, model not called
    no_interaction_found: Optional[bool] = None   # True: both drugs have FASS text, neither names the other


@app.get("/health")
def health():
    return {"status": "ok"}


def _decode_image(b64: Optional[str]) -> Optional[bytes]:
    if not b64:
        return None
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")


@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: AnswerRequest):
    image_bytes = _decode_image(req.image_b64)
    with _gpu_lock:
        try:
            return answer(text=req.text, image_bytes=image_bytes, state=req.state or {})
        except LLMUnavailable as e:        # hosted model quota/key/network problem -> the app shows "service down"
            raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    # Pay the model/index load time now, so the first user request is not slow
    # (the app times out after B_TIMEOUT seconds, default 120).
    import chatbot
    chatbot._load_model()          # no-op when LLM_BACKEND=groq
    chatbot._get_knowledge()
    uvicorn.run(app, host="0.0.0.0", port=8000)
