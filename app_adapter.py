"""Translate MediCheck's /answer response into the dict that app.py's render_result() expects.

Put `adapt()` where your teammate's src/b_client.py turns the HTTP response into the app's result
(or import this file there). API response  ->  app result:
    answer            -> explanation   (my disclaimer is stripped; app.py shows its own)
    severity|None     -> severity      (None -> INFO for normal answers, INSUFFICIENT_DATA for 'no FASS data')
    drugs [{name,..}] -> drugs         (list of display names)
    citations [url|label] -> citations [{source, section, url}]
    status 'refused'  -> safety_flag   ('crisis' or 'dose', set by the API)
    state             -> state         (app stores it and sends it back next turn)
"""
import re

DISCLAIMER_RE = re.compile(r"\s*This is not medical advice\. Please confirm with a pharmacist or doctor\.\s*$")
APP_DISCLAIMER = {
    "sv": "Detta är inte medicinsk rådgivning. Kontrollera med apotek eller läkare.",
    "en": "This is not medical advice. Please confirm with a pharmacist or doctor.",
    "ar": "هذه ليست نصيحة طبية. يرجى التأكد من الصيدلي أو الطبيب.",
}


def adapt(api: dict, lang: str = "en") -> dict:
    text = DISCLAIMER_RE.sub("", api.get("answer", "")).strip()
    status = api.get("status")
    flag = api.get("safety_flag")
    sev = api.get("severity")
    if not sev:
        if api.get("gap") or status in ("no_drug", "needs_confirmation"):
            sev = "INSUFFICIENT_DATA"
        else:
            sev = "INFO"   # incl. no_interaction_found: 'no documented interaction' is information, not a green light

    cites = []
    for c in api.get("citations", []) or []:
        is_url = str(c).startswith("http")
        cites.append({"source": "FASS", "section": "SmPC" if is_url else str(c),
                      "url": c if is_url else "https://fass.se"})
    return {
        "severity": sev,
        "explanation": text,
        "drugs": [d["name"] if isinstance(d, dict) else str(d) for d in api.get("drugs", [])],
        "citations": cites,
        "disclaimer": APP_DISCLAIMER.get(lang, APP_DISCLAIMER["en"]),
        "safety_flag": flag,
        "notes": [],
        "confidence": 0,            # the pipeline does not produce a confidence score; app hides 0
        "model": "b-api",
        "from_cache": False,
        "state": api.get("state"),
    }
