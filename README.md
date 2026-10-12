# MediCheck Sverige – Web Application

A chat-based web application built with Streamlit. Users ask about medicine interactions in
Swedish, English or Arabic, optionally with a photo of the packaging. The answer source is
selected with the `BACKEND` setting:

| `BACKEND` | Answer source | Purpose |
|---|---|---|
| `stub` (default) | Placeholder answers in `src/stub_backend.py` | Development, tests, offline demonstration |
| `b_api` | Fine-tuned Qwen model service, reached through ngrok (`POST /answer`) | Production use |

If the model service cannot be reached, the application shows a fixed "service unavailable"
notice. Answers to first questions are cached in `app_cache/`.

## 1. Running locally

```bash
python -m venv .venv
.venv\Scripts\activate                       # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt    # streamlit, requests, python-dotenv, pytest
copy .env.example .env                       # macOS/Linux: cp .env.example .env
```

Start the application:

    python -m streamlit run app.py

If Windows reports "Access is denied" for `C:\Users\<name>\.streamlit`, start it with:

    python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false

and open http://localhost:8501 in the browser.

Run the tests (no network access required):

    python -m pytest -q

## 2. Connecting the model service

1. In `.env`, set `BACKEND=b_api` and `B_API_URL=https://<service-name>.ngrok-free.dev`
   (and `B_API_KEY` if the service requires a key).
2. Check the connection with `python scripts/test_b_api.py`. The script reports the health
   status, response time, answer, detected severity label and returned state.
3. Start the application and ask, for example, "Can I take Waran with Ipren?".

Request sent to the service:
`{"text": <question> + <language instruction>, "image_b64": <photo or null>, "state": <state from the previous answer>}`

Expected response: `{"answer": "...", "state": ...}`. The severity badge is taken from a label
written in capital letters in the answer (`DANGEROUS`, `AVOID`, `CAUTION`, `INFO`). If no label
is found, the badge shows "Unknown".

## 3. Safety measures

- **Before the model is called:** messages indicating a crisis, a dosage question or an attempt
  to override the assistant's instructions receive fixed replies; the model is not called.
  Crisis replies include the emergency numbers 112 and 1177.
- **After the model answers:** the disclaimer is always a fixed text; answers containing
  dosage statements are replaced with a fixed notice; a pharmacist notice is always added,
  and a midwife/doctor/1177 notice is added when pregnancy is mentioned.
- **Logging:** `logs/requests.jsonl` records the label, model, response time and cache use.
  User messages are not logged.

The dosage check is intentionally strict and may also match harmless text such as
"Alvedon 500 mg". Such cases are recorded in `tests/challenger_cases.csv`.

## 4. Files

| File | Purpose |
|---|---|
| `app.py` | Chat interface: language selection, photo upload, severity badges, notices, disclaimer, right-to-left layout for Arabic |
| `src/medicheck.py` | Orchestrator: safety rules, cache, backend selection, post-model checks |
| `src/b_client.py` | Client for the model service (`/answer`, `/health`) |
| `src/safety.py` | Safety patterns, dosage detection, label extraction, fixed messages |
| `src/contracts.py` | Result format, labels, disclaimers and fixed notices (sv/en/ar) |
| `src/stub_backend.py` | Placeholder answers for six medicine pairs |
| `scripts/test_b_api.py` | Connection test for the model service |
| `tests/` | 78 automated tests and `challenger_cases.csv` (24 evaluation cases) |

## 5. Deployment (Streamlit Community Cloud)

1. Push the code to GitHub. Check with `git status` that `.env` is not included.
2. On share.streamlit.io, create an app from this repository with main file `app.py` and
   Python 3.11.
3. Enter the settings from `.streamlit/secrets.toml.example` under Secrets. Use
   `BACKEND = "stub"` until the model service is available, then `BACKEND = "b_api"`.
4. Open the application link and test it.

The deployed application returns model answers only while the model service and its ngrok
tunnel are running. Before a demonstration, the demonstration questions should be asked once
so that their answers are cached.
