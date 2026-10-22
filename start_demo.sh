#!/bin/bash
# MediCheck demo launcher for a fresh RunPod pod (no persistent storage needed).
#
#   bash start_demo.sh            # set everything up, start API + app, print the public link
#   QUANT=4bit bash start_demo.sh # same, 4-bit model for a 24-48 GB GPU
#   bash start_demo.sh stop       # stop everything
#   bash start_demo.sh status     # show what is running + the links again
#
# Run it from the repo root. Safe to run again: finished steps are skipped, running services are restarted.
# Optional: export NGROK_AUTHTOKEN=... to prefer ngrok over the Cloudflare quick tunnel (fallback).

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" || exit 1
LOGS="$REPO/logs"; RUN="$REPO/.run"; mkdir -p "$LOGS" "$RUN"
API_PORT=8000; APP_PORT=8501
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MEDICHECK_NO_CACHE=1
[ -n "$QUANT" ] && export MEDICHECK_QUANT="$QUANT"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mOK\033[0m  %s\n' "$*"; }
warn() { printf '   \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*"; exit 1; }

stop_all() {
  for n in api app tunnel; do
    [ -f "$RUN/$n.pid" ] && kill "$(cat "$RUN/$n.pid")" 2>/dev/null; rm -f "$RUN/$n.pid"
  done
  pkill -f "api.py" 2>/dev/null; pkill -f "streamlit run app.py" 2>/dev/null
  pkill -f "cloudflared tunnel" 2>/dev/null; pkill -f "ngrok http" 2>/dev/null
  sleep 2
}

show_links() {
  say "LINKS"
  [ -f "$RUN/public_url" ] && printf '   PUBLIC APP URL : \033[1;32m%s\033[0m\n' "$(cat "$RUN/public_url")" \
                           || warn "no public URL recorded (tunnel not started)"
  echo   "   API health     : curl localhost:$API_PORT/health"
  echo   "   logs           : tail -f $LOGS/api.log   |   $LOGS/app.log   |   $LOGS/tunnel.log"
  echo   "   stop           : bash start_demo.sh stop"
}

case "$1" in
  stop)   say "Stopping"; stop_all; ok "stopped"; exit 0 ;;
  status) say "Status"
          curl -s localhost:$API_PORT/health && echo || warn "API not answering"
          curl -sI localhost:$APP_PORT | head -1 || warn "app not answering"
          show_links; exit 0 ;;
esac

START=$(date +%s)
say "1/8  GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no GPU visible (nvidia-smi failed)"

say "2/8  Code"
if [ -d .git ]; then git pull --ff-only 2>&1 | tail -1 || warn "git pull skipped"; fi
for f in api.py chatbot.py app.py src/medicheck.py intents.py llm_ayesha/fass_sections.csv; do
  [ -f "$f" ] || die "missing $f - did you push all the fixed files to GitHub?"
done
ok "files present"

say "3/8  Python packages"
if python -c "import chromadb, sentence_transformers, easyocr, fastapi, streamlit, dotenv, rapidfuzz, langdetect" 2>/dev/null; then
  ok "already installed"
else
  pip install -q -r requirements.txt fastapi uvicorn python-multipart langdetect python-dotenv \
    || die "pip install failed"
  ok "installed"
fi

say "4/8  .env"
cat > .env <<ENV
BACKEND=b_api
B_API_URL=http://localhost:$API_PORT
B_API_KEY=
ENV
ok ".env written (BACKEND=b_api, B_API_URL=http://localhost:$API_PORT)"

say "5/8  Databases (rebuilt from the CSVs; the repo's drug_db was only a Git-LFS placeholder)"
DBSIZE=$(stat -c %s llm_ayesha/drug_db/chroma.sqlite3 2>/dev/null || echo 0)
if [ "$DBSIZE" -gt 1000000 ]; then
  ok "database already built"
else
  rm -rf llm_ayesha/drug_db
  ( cd llm_ayesha && python build_chromadb.py ) > "$LOGS/build_chroma.log" 2>&1 \
    && python build_sections_db.py > "$LOGS/build_sections.log" 2>&1 \
    && ok "ChromaDB built" \
    || warn "ChromaDB build failed (see logs/build_*.log) - the app falls back to the CSV files, answers still work"
fi

say "6/8  Models (language model ~60 GB + OCR models; first run only)"
python - <<'PY' || die "model download failed"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-30B-A3B-Instruct-2507")
print("   language model: ready")
PY
python -c "import easyocr; easyocr.Reader(['sv','en'], gpu=True, verbose=False); print('   OCR models: ready')" 2>&1 | tail -1

say "7/8  Start API and app"
stop_all
nohup python api.py > "$LOGS/api.log" 2>&1 &
echo $! > "$RUN/api.pid"
echo "   waiting for the model to load (3-8 minutes the first time)..."
for i in $(seq 1 180); do
  if curl -sf localhost:$API_PORT/health >/dev/null 2>&1; then ok "API is up after ~$((i*5))s"; break; fi
  kill -0 "$(cat "$RUN/api.pid")" 2>/dev/null || { tail -30 "$LOGS/api.log"; die "api.py crashed (log above)"; }
  sleep 5
  [ "$i" = 180 ] && { tail -30 "$LOGS/api.log"; die "API did not come up in 15 minutes"; }
done
echo "   warm-up question..."
R=$(curl -s -m 180 -X POST localhost:$API_PORT/answer -H 'content-type: application/json' \
      -d '{"text":"Can I take erytromycin together with flukonazol?","state":null}')
echo "$R" | python -c "import sys,json; d=json.load(sys.stdin); print('   severity:',d.get('severity'),'| citations:',len(d.get('citation_details') or []))" \
  || warn "warm-up gave no valid answer: ${R:0:200}"
nohup streamlit run app.py --server.address 0.0.0.0 --server.port $APP_PORT --server.headless true \
      --server.enableCORS false --server.enableXsrfProtection false > "$LOGS/app.log" 2>&1 &
echo $! > "$RUN/app.pid"
for i in $(seq 1 30); do curl -sf localhost:$APP_PORT >/dev/null 2>&1 && { ok "Streamlit is up"; break; }; sleep 2; done
curl -sf localhost:$APP_PORT >/dev/null 2>&1 || { tail -20 "$LOGS/app.log"; die "Streamlit did not start"; }

say "8/8  Public link"
rm -f "$RUN/public_url"
# a) RunPod's own proxy: works when port $APP_PORT is exposed on the pod
if [ -n "$RUNPOD_POD_ID" ]; then
  RP="https://${RUNPOD_POD_ID}-${APP_PORT}.proxy.runpod.net"
  CODE=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$RP")
  if [ "$CODE" = "200" ] || [ "$CODE" = "302" ]; then echo "$RP" > "$RUN/public_url"; ok "RunPod proxy works"; else warn "RunPod proxy on $APP_PORT not reachable (HTTP $CODE) - expose port $APP_PORT on the pod, or use the tunnel below"; fi
fi
# b) ngrok, if you gave a token
if [ ! -f "$RUN/public_url" ] && [ -n "$NGROK_AUTHTOKEN" ]; then
  command -v ngrok >/dev/null || { curl -sL https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-amd64.tgz | tar xz -C /usr/local/bin 2>/dev/null; }
  if command -v ngrok >/dev/null; then
    ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1
    nohup ngrok http $APP_PORT --log=stdout > "$LOGS/tunnel.log" 2>&1 & echo $! > "$RUN/tunnel.pid"
    for i in $(seq 1 20); do
      U=$(curl -s localhost:4040/api/tunnels | python -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)
      [ -n "$U" ] && { echo "$U" > "$RUN/public_url"; ok "ngrok tunnel"; break; }; sleep 2
    done
  fi
fi
# c) Cloudflare quick tunnel: no account or token needed
if [ ! -f "$RUN/public_url" ]; then
  [ -x "$RUN/cloudflared" ] || curl -sL -o "$RUN/cloudflared" \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x "$RUN/cloudflared"
  if [ -x "$RUN/cloudflared" ]; then
    nohup "$RUN/cloudflared" tunnel --url http://localhost:$APP_PORT --no-autoupdate > "$LOGS/tunnel.log" 2>&1 & echo $! > "$RUN/tunnel.pid"
    for i in $(seq 1 30); do
      U=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOGS/tunnel.log" | head -1)
      [ -n "$U" ] && { echo "$U" > "$RUN/public_url"; ok "Cloudflare tunnel"; break; }; sleep 2
    done
  fi
fi

say "DONE in $(( ($(date +%s) - START) / 60 )) min"
show_links
echo
echo "   Open the PUBLIC APP URL, check the sidebar says 'b_api' and 'Test connection' = Connected."
echo "   Tunnel pages can take ~10 s to start answering; reload once if it shows an error."
