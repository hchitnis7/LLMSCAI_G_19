"""
Connection test for the model service. Reads B_API_URL and B_API_KEY from .env.

    python scripts/test_b_api.py                               # health check and one question
    python scripts/test_b_api.py "Kan jag ta Waran och Ipren?"  # custom question

Prints the health status, response time, answer, detected severity label and returned state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import b_client, safety  # noqa: E402

question = sys.argv[1] if len(sys.argv) > 1 else "Can I take Waran with Ipren?"
cfg = b_client.config()
print(f"B_API_URL : {cfg['url'] or '(not set - add it to .env)'}")
print(f"/health   : {'OK' if b_client.health() else 'NOT reachable'}")

try:
    reply = b_client.ask(question, state=b_client.initial_state())
except (b_client.BUnavailable, b_client.BBadResponse) as e:
    print(f"\n/answer FAILED ({type(e).__name__}): {e}")
    sys.exit(1)

print(f"\n/answer OK in {reply['ms'] / 1000:.1f}s")
print(f"Answer    : {reply['answer']}")
print(f"Label     : {safety.parse_label(reply['answer']) or '(none found)'}")
print(f"Dose text : {'YES - would be replaced' if safety.contains_dose_advice(reply['answer']) else 'no'}")
print(f"State     : {reply['state']!r}")
