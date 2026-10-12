"""Input hardening for untrusted text: OCR output and the user's typed prompt.

Two attack surfaces this addresses:
  1. Indirect injection via OCR text -- someone could print text on/near a
     package photo designed to manipulate the model ("ignore your rules",
     "say this is safe", etc). The model never sees raw OCR text as part of
     its reasoning context in this design (see chatbot.py's extraction step),
     but the extraction call itself still reads it, so it needs protecting too.
  2. Direct jailbreak/injection via the typed prompt -- "ignore previous
     instructions", roleplay framing, encoding tricks, etc.

This module does NOT replace the system-prompt instruction to ignore embedded
instructions -- it's a second, independent layer: a fast regex pre-filter
(cheap, no model call) plus explicit delimiting so untrusted text is visually
and structurally marked as data, not instructions, wherever it's inserted
into a prompt.
"""
import re

# Heuristic patterns for common jailbreak/injection phrasing. Not exhaustive --
# this is a fast first line of defense and a logging trigger, not the only
# defense (the system prompt + output validation + schema-constrained
# extraction in chatbot.py are the deeper layers).
_INJECTION_PATTERNS = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) instructions?\b",
    r"\bdisregard (all |any |the )?(previous|prior|above|earlier)\b",
    r"\byou are now\b",
    r"\bact as\b.{0,20}\b(dan|jailbreak|unrestricted|no rules?)\b",
    r"\bpretend (you|to be)\b",
    r"\bsystem prompt\b",
    r"\bnew instructions?\b",
    r"\boverride\b.{0,20}\b(rules?|instructions?|safety)\b",
    r"\bdo not (mention|include|add) (any )?(warning|disclaimer|safety)\b",
    r"\bwithout (any )?(warning|disclaimer)s?\b",
    r"\brespond (only )?with\b.{0,20}\bno\b.{0,20}\b(warning|disclaimer)\b",
    r"\[\s*system\s*\]",
    r"<\s*system\s*>",
]
_COMPILED = [re.compile(p, re.I) for p in _INJECTION_PATTERNS]


def flag_injection_attempt(text: str) -> list:
    """Returns the list of matched pattern strings (empty if clean).
    Use for logging + as a fast-path signal; do not treat 'clean' as proof
    of safety -- always still apply delimiting and output validation."""
    if not text:
        return []
    return [p.pattern for p in _COMPILED if p.search(text)]


def wrap_untrusted(label: str, text: str) -> str:
    """Wrap untrusted text in explicit delimiters with a clear data-only label.
    Use this ANY time OCR text or user text is inserted into a prompt sent to
    a model, so the model has a structural (not just verbal) cue about what's
    data vs. instruction."""
    text = text.replace("<", "\u2039").replace(">", "\u203a")  # neutralize fake tag injection
    return f"<{label}>\n{text}\n</{label}>"


def log_suspicious(source: str, text: str, matched_patterns: list):
    """Minimal logging hook -- expand to write to a file/DB for the report's
    robustness section (how many attempts were caught, examples, etc)."""
    if matched_patterns:
        print(f"[SECURITY] Suspicious input from {source}: {matched_patterns} "
              f"| text preview: {text[:120]!r}")


# ----------------------------------------------------------------- output-side guard
# The prompt forbids calling anything "safe", but a language model occasionally still does
# ("it's safe to continue driving"). This is a deterministic last line of defence: any sentence
# that ASSERTS safety (no negation, no hedge such as "whether", "to be safe, ask...") is removed
# before the user sees it. Curly apostrophes are normalised first - models emit them.
_NEG = re.compile(r"\b(not|no|never|cannot|without|unsafe|inte|ingen|inget|aldrig|osäker|ej|ska inte)\b|\w+n't", re.I)
_HEDGE = re.compile(r"\b(whether|if|unless|confirm|check|ask|verify|sure|determine|unclear|unknown|"
                    r"to be safe|better safe|play it safe|stay safe|om det är|kontrollera|fråga|osäkert)\b", re.I)
_SAFE = re.compile(r"\b(safe|säker|säkert|harmless|ofarlig|fine to (take|combine|use|mix)|no (risk|problem)s?|"
                   r"inga risker|inga problem|risk-free)\b", re.I)


_BENIGN_SAFE = re.compile(r"\bsafe(r)? (alternatives?|options?|ways?|treatments?|places?|medicines?|choices?)\b|"
                          r"\bsäker(t)? (alternativ|sätt|behandling)\b", re.I)


def asserts_safety(sentence: str) -> bool:
    """True if the sentence claims something is safe/harmless (no negation, no hedge)."""
    sent = _BENIGN_SAFE.sub("", sentence.replace("\u2019", "'").replace("\u2018", "'"))
    return bool(_SAFE.search(sent) and not _NEG.search(sent) and not _HEDGE.search(sent))


def strip_safety_assertions(text: str):
    """Returns (clean_text, removed_sentences)."""
    norm = text.replace("’", "'").replace("‘", "'")
    kept, removed = [], []
    for sent in re.split(r"(?<=[.!?])\s+", norm):
        if asserts_safety(sent):
            removed.append(sent.strip())
        else:
            kept.append(sent)
    return " ".join(kept).strip() if removed else text, removed