"""Reference for Ida (UI/deployment): calling the locally-hosted model over
HTTP instead of importing chatbot.py directly (chatbot.py needs a GPU that
Streamlit Cloud doesn't have -- the model runs on a teammate's machine and
is exposed via a tunnel URL).

Replace API_URL with the ngrok/cloudflared URL you're given each session.
"""
import base64

import requests
import streamlit as st

API_URL = "https://overexert-earflap-twelve.ngrok-free.dev"  # <-- update this
TIMEOUT = 120  # seconds; model generation can take a few seconds, be generous

st.title("MediCheck Sverige")

# Check the backend is reachable before showing the chat
try:
    requests.get(f"{API_URL}/health", timeout=5)
    backend_up = True
except requests.exceptions.RequestException:
    backend_up = False
    st.warning("The medication-checking backend is currently offline. "
              "Please let the team know, or try again shortly.")

if "med_state" not in st.session_state:
    st.session_state.med_state = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("image"):
            st.image(m["image"], width=220)
        st.write(m["content"])

# --- Input: text + image in ONE message (Streamlit >= 1.43), like a normal chat app.
# Falls back to a separate uploader on older Streamlit versions.
text, image_bytes = None, None
try:
    submitted = st.chat_input(
        "Ask about a medication, or attach a photo of the package...",
        accept_file=True,
        file_type=["jpg", "jpeg", "png"],
        disabled=not backend_up,
    )
    if submitted:
        text = (submitted.text or "").strip()
        if submitted.files:
            image_bytes = submitted.files[0].getvalue()   # getvalue(), NOT read()
except TypeError:
    # Old Streamlit: no accept_file. Use uploader, but send each image only ONCE
    # (otherwise it is re-sent with every later message, e.g. a plain "yes").
    image_file = st.file_uploader("Upload a photo of the medication (optional)",
                                  type=["jpg", "jpeg", "png"])
    prompt = st.chat_input("Ask about a medication...", disabled=not backend_up)
    if prompt:
        text = prompt.strip()
        if image_file is not None and st.session_state.get("last_image_id") != image_file.file_id:
            image_bytes = image_file.getvalue()           # getvalue(), NOT read()
            st.session_state.last_image_id = image_file.file_id

if text or image_bytes:
    st.session_state.messages.append({
        "role": "user",
        "content": text or "(photo only)",
        "image": image_bytes,
    })

    image_b64 = base64.b64encode(image_bytes).decode() if image_bytes else None

    with st.spinner("Checking..."):
        try:
            resp = requests.post(
                f"{API_URL}/answer",
                json={"text": text or "", "image_b64": image_b64, "state": st.session_state.med_state},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json()
            st.session_state.med_state = result["state"]
            reply = result["answer"]
        except requests.exceptions.RequestException as e:
            reply = f"Sorry, I couldn't reach the backend right now. ({e})"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()