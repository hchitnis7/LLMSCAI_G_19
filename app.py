"""
MediCheck Sverige: chat interface built with Streamlit.

Run locally:  python -m streamlit run app.py

The interface communicates only with src.medicheck.chat() and never calls a model directly.
"""
from __future__ import annotations

import html

import streamlit as st

from src import medicheck
from src.contracts import LOW_CONFIDENCE

st.set_page_config(page_title="MediCheck Sverige", page_icon="💊", layout="centered")

# ---------------- Interface texts ----------------
T = {
    "sv": dict(
        title="MediCheck Sverige", tagline="Fråga om dina läkemedel kan tas tillsammans.",
        ai_notice="Jag är en AI, inte en farmaceut.",
        placeholder="t.ex. Kan jag ta Waran och Ipren?",
        examples="Prova: *Kan jag ta Waran och Ipren?* · *Alvedon med alkohol?*",
        photo_label="Bifoga foto av förpackning (JPG/PNG)", photo_note="Fotot skickas med ditt nästa meddelande.",
        new_chat="Ny konversation", thinking="Tänker…", photo_sent="📷 foto bifogat",
        photo_found="Läst från fotot", photo_low="Fotot var svårt att läsa. Skriv gärna läkemedlets namn.",
        sources="Källor", db_date="Databasen uppdaterad", cached="Sparat svar", confidence="Säkerhet",
        answered_by="Svar från", error="Något gick fel. Försök igen, eller fråga ditt apotek.",
        stub="TESTLÄGE: exempeldata, inte riktiga svar.", crisis_title="Du är inte ensam",
        status="Svarstjänst", check_conn="Testa anslutning", conn_ok="Ansluten", conn_bad="Når inte tjänsten",
    ),
    "en": dict(
        title="MediCheck Sverige", tagline="Ask whether your medicines can be taken together.",
        ai_notice="I am an AI, not a pharmacist.",
        placeholder="e.g. Can I take Waran with Ipren?",
        examples="Try: *Can I take Waran with Ipren?* · *Alvedon with alcohol?*",
        photo_label="Attach a photo of the packaging (JPG/PNG)", photo_note="The photo is sent with your next message.",
        new_chat="New conversation", thinking="Thinking…", photo_sent="📷 photo attached",
        photo_found="Read from the photo", photo_low="The photo was hard to read. Please type the medicine name.",
        sources="Sources", db_date="Database updated", cached="Saved answer", confidence="Confidence",
        answered_by="Answered by", error="Something went wrong. Please try again, or ask your pharmacy.",
        stub="TEST MODE: sample data, not real answers.", crisis_title="You are not alone",
        status="Answer service", check_conn="Test connection", conn_ok="Connected", conn_bad="Can't reach the service",
    ),
    "ar": dict(
        title="MediCheck Sverige", tagline="اسأل عما إذا كان يمكن تناول أدويتك معًا.",
        ai_notice="أنا ذكاء اصطناعي، ولست صيدليًا.",
        placeholder="مثال: هل يمكنني تناول Waran مع Ipren؟",
        examples="جرّب: *هل يمكنني تناول Waran مع Ipren؟*",
        photo_label="أرفق صورة للعلبة (JPG/PNG)", photo_note="ستُرسل الصورة مع رسالتك التالية.",
        new_chat="محادثة جديدة", thinking="جارٍ التفكير…", photo_sent="📷 صورة مرفقة",
        photo_found="تمت قراءته من الصورة", photo_low="كان من الصعب قراءة الصورة. اكتب اسم الدواء من فضلك.",
        sources="المصادر", db_date="آخر تحديث لقاعدة البيانات", cached="إجابة محفوظة", confidence="الثقة",
        answered_by="الإجابة من", error="حدث خطأ. حاول مرة أخرى أو اسأل الصيدلية.",
        stub="وضع الاختبار: بيانات تجريبية وليست إجابات حقيقية.", crisis_title="لست وحدك",
        status="خدمة الإجابات", check_conn="اختبار الاتصال", conn_ok="متصل", conn_bad="تعذر الوصول إلى الخدمة",
    ),
}

SEVERITY_STYLE = {  # label per language, background colour, text colour
    "DANGEROUS": ({"sv": "FARLIGT", "en": "DANGEROUS", "ar": "خطير"}, "#b3261e", "#fff"),
    "AVOID": ({"sv": "UNDVIK", "en": "AVOID", "ar": "تجنّب"}, "#e8710a", "#fff"),
    "CAUTION": ({"sv": "FÖRSIKTIGHET", "en": "CAUTION", "ar": "بحذر"}, "#f9ab00", "#1f1f1f"),
    "INFO": ({"sv": "INFORMATION", "en": "INFORMATION", "ar": "معلومة"}, "#1a73e8", "#fff"),
    "SAFE": ({"sv": "INGEN KÄND RISK", "en": "NO KNOWN RISK", "ar": "لا خطر معروف"}, "#188038", "#fff"),
    "INSUFFICIENT_DATA": ({"sv": "OKÄND", "en": "UNKNOWN", "ar": "غير معروف"}, "#5f6368", "#fff"),
}

MODEL_NAMES = {"stub": {"sv": "testdata", "en": "test data", "ar": "بيانات تجريبية"},
               "b-api": {"sv": "MediCheck-modellen", "en": "the MediCheck model", "ar": "نموذج MediCheck"}}

LANG_NAMES = {"sv": "Svenska", "en": "English", "ar": "العربية"}

# ---------------- Session state ----------------
st.session_state.setdefault("messages", [])     # conversation history
st.session_state.setdefault("conv_state", None)  # state returned by the model service
st.session_state.setdefault("uploader_key", 0)   # changed to reset the photo uploader

# ---------------- Sidebar ----------------
lang = st.sidebar.radio("Språk / Language / اللغة", list(LANG_NAMES), format_func=LANG_NAMES.get)
t = T[lang]

if st.sidebar.button(t["new_chat"]):
    st.session_state.messages = []
    st.session_state.conv_state = None
    st.session_state.uploader_key += 1
    st.rerun()

photo = st.sidebar.file_uploader(t["photo_label"], type=["jpg", "jpeg", "png"],
                                 key=f"photo_{st.session_state.uploader_key}")
if photo is not None:
    st.sidebar.image(photo, width=180)
    st.sidebar.caption(t["photo_note"])

backend = medicheck.backend_name()
st.sidebar.divider()
st.sidebar.caption(f"{t['status']}: `{backend}`")
if backend == "b_api" and st.sidebar.button(t["check_conn"]):
    from src import b_client
    if b_client.health():
        st.sidebar.success(t["conn_ok"])
    else:
        st.sidebar.error(t["conn_bad"])

if lang == "ar":  # right-to-left layout for Arabic
    st.markdown("<style>[data-testid='stMainBlockContainer'], .main .block-container"
                "{direction:rtl;text-align:right}</style>", unsafe_allow_html=True)

# ---------------- Header ----------------
st.title(t["title"])
st.caption(f"{t['tagline']} · **{t['ai_notice']}**")
if medicheck.use_stub():
    st.warning(t["stub"])


# ---------------- Result display ----------------
def badge(severity: str) -> str:
    labels, bg, fg = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["INSUFFICIENT_DATA"])
    return (f"<span style='background:{bg};color:{fg};padding:4px 12px;border-radius:8px;"
            f"font-weight:700'>{html.escape(labels[lang])}</span>")


def render_result(res: dict) -> None:
    flag = res.get("safety_flag")
    if flag == "crisis":
        st.error(f"**{t['crisis_title']}**\n\n{res['explanation']}")
    elif flag in ("dose", "injection", "backend_down"):
        st.info(res["explanation"])
    else:
        st.markdown(badge(res["severity"]), unsafe_allow_html=True)
        if res.get("drugs"):
            st.markdown(" · ".join(f"**{html.escape(d)}**" for d in res["drugs"]))
        ocr = res.get("ocr")
        if ocr:
            if ocr.get("names"):
                st.caption(f"{t['photo_found']}: {', '.join(ocr['names'])}")
            if ocr.get("confidence", 1) < LOW_CONFIDENCE:
                st.warning(t["photo_low"])
        if flag == "dose_in_answer":
            st.info(res["explanation"])
        else:
            st.write(res["explanation"])
        if res.get("citations"):
            st.markdown(f"**{t['sources']}**")
            for c in res["citations"]:
                if c.get("url"):
                    st.markdown(f"- [{c['source']}: {c['section']}]({c['url']})")
                else:
                    st.markdown(f"- {c['source']}: {c['section']}")
        for note in res.get("notes", []):
            st.markdown(f"- {note}")
        meta = []
        if res.get("confidence", 0) > 0:
            meta.append(f"{t['confidence']}: {res['confidence']:.0%}")
        if res.get("db_last_updated"):
            meta.append(f"{t['db_date']}: {res['db_last_updated']}")
        if res.get("model") in MODEL_NAMES:
            meta.append(f"{t['answered_by']}: {MODEL_NAMES[res['model']][lang]}")
        if res.get("from_cache"):
            meta.append(t["cached"])
        if meta:
            st.caption(" · ".join(meta))
    st.caption(f"⚕️ {res['disclaimer']}")  # shown with every answer


# ---------------- Conversation ----------------
if not st.session_state.messages:
    st.caption(t["examples"])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        elif msg.get("result"):
            render_result(msg["result"])
        else:
            st.error(msg["content"])

prompt = st.chat_input(t["placeholder"])
if prompt:
    image_bytes = photo.getvalue() if photo is not None else None
    mime = (photo.type or "image/jpeg") if photo is not None else "image/jpeg"
    shown = prompt + (f"  \n_{t['photo_sent']}_" if image_bytes else "")
    st.session_state.messages.append({"role": "user", "content": shown})
    with st.chat_message("user"):
        st.write(shown)

    with st.chat_message("assistant"):
        try:
            with st.spinner(t["thinking"]):
                res = medicheck.chat(prompt, lang, st.session_state.conv_state, image_bytes, mime)
            st.session_state.conv_state = res.get("state")
            render_result(res)
            st.session_state.messages.append({"role": "assistant", "content": "", "result": res})
        except Exception:
            st.error(t["error"])
            st.session_state.messages.append({"role": "assistant", "content": t["error"]})

    if image_bytes:  # a photo is attached to one message only
        st.session_state.uploader_key += 1
        st.rerun()
