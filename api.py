"""Local API server wrapping chatbot.answer(), for Ida (UI/deployment) to call
over the network since the model runs on your machine, not on Streamlit Cloud.

Run:
    pip install fastapi uvicorn python-multipart
    python api.py
Then expose it publicly (pick one):
    ngrok http 8000
    cloudflared tunnel --url http://localhost:8000
Send the printed https URL to your teammate. It changes each time you
restart the tunnel unless you set up a reserved/static ngrok domain.
"""
import base64
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from chatbot import answer

app = FastAPI(title="MediCheck Sverige API")


class AnswerRequest(BaseModel):
    text: str = ""
    image_b64: Optional[str] = None   # base64-encoded image bytes, or null
    state: dict = {}


class AnswerResponse(BaseModel):
    status: str
    answer: str
    drugs: list
    severity: Optional[str]
    citations: list
    state: dict
    safety_flag: Optional[str] = None   # 'crisis' | 'dose' | None
    gap: Optional[bool] = None                    # True: fixed 'no FASS data' reply, model not called
    no_interaction_found: Optional[bool] = None   # True: both drugs have FASS text, neither names the other


@app.get("/health")
def health():
    """Teammate's app should call this first to check the server is up
    before showing the chat UI, and can poll it to show a 'bot is offline'
    message if it stops responding."""
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: AnswerRequest):
    image_bytes = base64.b64decode(req.image_b64) if req.image_b64 else None
    result = answer(text=req.text, image_bytes=image_bytes, state=req.state)
    return result


if __name__ == "__main__":
    # Loading the model happens on the FIRST /answer call (lazy loading in
    # chatbot.py), not at startup, so the server starts instantly but the
    # first real request will be slow (model load time). Consider calling
    # answer(text="warm up") once here if you want load time paid up front.
    uvicorn.run(app, host="0.0.0.0", port=8000)