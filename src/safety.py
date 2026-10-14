"""
Safety rules applied before and after the model is called.

Before the model: messages indicating a crisis, a dosage question or an attempt to change
the assistant's instructions are answered with fixed texts, and the model is not called.
After the model: answers are checked for dosage statements, and the severity label is
extracted from the answer text.

The pattern lists cover Swedish, English and Arabic. They are intentionally simple and
should be extended when evaluation identifies phrasings that are not recognised.
"""
from __future__ import annotations

import re

_CRISIS = [
    # English
    r"suicide", r"kill myself", r"end my life", r"want to die", r"hurt myself", r"self[- ]?harm",
    r"take (a lot|lots|all) of (my )?pills", r"overdose",
    # Swedish
    r"självmord", r"ta livet av mig", r"vill dö", r"skada mig själv", r"överdos",
    r"ta (en massa|många|alla) (tabletter|piller)",
    # Arabic
    r"انتحار", r"أقتل نفسي", r"اقتل نفسي", r"أريد أن أموت", r"جرعة زائدة", r"أؤذي نفسي",
]

_DOSE = [
    r"\bdos(e|es|ing|age)?\b", r"how (much|many) (should|can|do) i take", r"\bmg\b",
    r"\bdosering\b", r"hur (mycket|många) (ska|kan|får) jag ta",
    r"جرعة", r"كم (حبة|قرص|ملغ)",
]

_INJECTION = [
    r"ignore (all |the |your )?(previous |prior )?instructions", r"system prompt",
    r"you are now", r"diagnos(e|is) me", r"act as (a|my) (doctor|pharmacist)",
    r"ignorera (alla |dina )?(tidigare )?instruktioner", r"ställ en diagnos",
    r"تجاهل (كل |جميع )?التعليمات",
]


# ---------- Checks applied to the model's answer ----------
# These checks are deliberately strict: an answer that states an amount or an instruction
# such as "take two tablets" is replaced with a fixed message. Cases where this is too
# strict are recorded during evaluation and the patterns adjusted accordingly.
_DOSE_IN_ANSWER = [
    r"\b\d+([.,]\d+)?\s*(mg|milligram|g|gram|ml|mikrogram|µg|mcg)\b",
    r"\b\d+([.,]\d+)?\s*(ملغ|مغ|مليغرام|غرام|مل)",
    r"\btake\s+(\d+|one|two|three|four|1-2|1–2)\s+(tablet|tablets|pill|pills|capsule|capsules)\b",
    r"\bta\s+(\d+|en|ett|två|tre|fyra|1-2|1–2)\s+(tablett|tabletter|kapsel|kapslar)\b",
    r"\b(\d+|one|two|three|four)\s+times\s+(a|per)\s+day\b",
    r"\b(\d+|en|två|tre|fyra)\s+gånger\s+(om|per)\s+dag(en)?\b",
]

_PREGNANCY = [r"\bpreg\w*", r"\bgravid\w*", r"حامل", r"الحمل"]

_LABEL_RE = re.compile(r"\b(DANGEROUS|AVOID|CAUTION|INFO|FARLIGT|FARLIG|UNDVIK|FÖRSIKTIGHET)\b")
_LABEL_MAP = {"FARLIG": "DANGEROUS", "FARLIGT": "DANGEROUS", "UNDVIK": "AVOID",
              "FÖRSIKTIGHET": "CAUTION"}
_LABEL_PREFIX_RE = re.compile(
    r"^\s*[\[\(*]*\s*(?:severity|label|risk|allvarlighetsgrad|nivå)?\s*[:=]?\s*[\[\(*]*\s*"
    r"(DANGEROUS|AVOID|CAUTION|INFO|FARLIGT|FARLIG|UNDVIK|FÖRSIKTIGHET)\s*[\]\)*]*\s*[:\-–—.]*\s*",
    re.IGNORECASE)


def contains_dose_advice(answer: str) -> bool:
    """Return True if the answer states an amount or a dosing instruction."""
    t = (answer or "").lower()
    return any(re.search(p, t) for p in _DOSE_IN_ANSWER)


def mentions_pregnancy(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _PREGNANCY)


def parse_label(answer: str) -> str | None:
    """Return the severity label written in capital letters in the answer, or None.

    Only upper-case labels are recognised, so that ordinary wording such as
    "not dangerous" does not determine the displayed severity.
    """
    m = _LABEL_RE.search(answer or "")
    if not m:
        return None
    return _LABEL_MAP.get(m.group(1), m.group(1))


def strip_label_prefix(answer: str) -> str:
    """Remove a leading severity label, which is displayed separately in the interface."""
    return _LABEL_PREFIX_RE.sub("", answer or "", count=1).strip() or (answer or "").strip()


def _match(patterns: list[str], text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def classify(text: str) -> str | None:
    """Return "crisis", "injection", "dose" or None. Crisis takes precedence."""
    if not text:
        return None
    if _match(_CRISIS, text):
        return "crisis"
    if _match(_INJECTION, text):
        return "injection"
    if _match(_DOSE, text):
        return "dose"
    return None


MESSAGES = {
    "crisis": {
        "en": ("It sounds like you may be going through something very hard. You don't have to handle it alone. "
               "If you are in immediate danger, call 112 now. For support, call 1177 or Mind Självmordslinjen on 90101."),
        "sv": ("Det låter som att du har det väldigt tungt just nu. Du behöver inte hantera det ensam. "
               "Är du i akut fara, ring 112 nu. För stöd, ring 1177 eller Mind Självmordslinjen på 90101."),
        "ar": ("يبدو أنك تمر بوقت صعب جدًا. لست مضطرًا لمواجهة ذلك وحدك. "
               "إذا كنت في خطر فوري، اتصل بالرقم 112 الآن. للدعم اتصل بـ 1177 أو خط Mind على الرقم 90101."),
    },
    "dose": {
        "en": "MediCheck can't give dosing advice. The right dose depends on you; please ask your pharmacist or doctor, or check the package leaflet.",
        "sv": "MediCheck kan inte ge råd om dosering. Rätt dos beror på dig; fråga ditt apotek eller din läkare, eller läs bipacksedeln.",
        "ar": "لا يستطيع MediCheck تقديم نصائح حول الجرعات. الجرعة المناسبة تعتمد عليك؛ اسأل الصيدلي أو الطبيب أو راجع النشرة الداخلية.",
    },
    "dose_in_answer": {
        "en": "The answer contained dosing details, which MediCheck doesn't give. Please ask your pharmacist or doctor about the right dose, or check the package leaflet.",
        "sv": "Svaret innehöll doseringsuppgifter, vilket MediCheck inte ger. Fråga ditt apotek eller din läkare om rätt dos, eller läs bipacksedeln.",
        "ar": "احتوت الإجابة على تفاصيل جرعات، وهو ما لا يقدمه MediCheck. اسأل الصيدلي أو الطبيب عن الجرعة المناسبة أو راجع النشرة الداخلية.",
    },
    "backend_down": {
        "en": "MediCheck can't reach its answer service right now. Please try again in a few minutes, or ask your pharmacist.",
        "sv": "MediCheck når inte sin svarstjänst just nu. Försök igen om några minuter, eller fråga ditt apotek.",
        "ar": "لا يستطيع MediCheck الوصول إلى خدمة الإجابات الآن. حاول مرة أخرى بعد بضع دقائق أو اسأل الصيدلي.",
    },
    "injection": {
        "en": "MediCheck only checks medication interactions using Swedish sources. It can't diagnose or change how it works. Please see a doctor or call 1177 for medical advice.",
        "sv": "MediCheck kontrollerar bara läkemedelsinteraktioner utifrån svenska källor. Den kan inte ställa diagnoser eller ändra hur den fungerar. Kontakta läkare eller ring 1177 för medicinsk rådgivning.",
        "ar": "يتحقق MediCheck فقط من التداخلات الدوائية باستخدام مصادر سويدية. لا يمكنه التشخيص أو تغيير طريقة عمله. راجع طبيبًا أو اتصل بـ 1177 للحصول على استشارة طبية.",
    },
}
