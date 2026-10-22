"""What is the user actually asking? Keyword-based, same deliberate style as
topics.py and prompts.py's language detection -- fast, deterministic, no
extra model call. Returns a list of intent keys matching rag_prompts.py's
INTENT_LABELS, so the result plugs straight into build_user_block().

More than one intent can be true at once ("is it safe to drink on this, and
what's it even for?" -> ["topic", "indication"]), and rag_chat handles each.
"""
import re

_PATTERNS = {
    "indication": [
        r"\bwhat('?s| is) (it|this|\w+) (for|used for|prescribed for)\b",
        r"\bwhy (do|would) i take\b", r"\bwhat does .* treat\b",
        r"\bvad (är|används) .* för\b", r"\bvarför tar jag\b",
        r"\bvad används .* (mot|till)\b",
        # "is X the same as Y" / "difference between X and Y" -> what is it
        r"\b(is|are)\b.*\b(same|equivalent)\b.*\b(as|to)\b", r"\bsame (thing |drug |medicine )?as\b",
        r"\bdifference between\b", r"\bär .* samma (sak )?som\b", r"\bskillnad(en)? mellan\b",
        # bare "what is X" / "tell me about X" / "vad är X" (one or two words after) = what is it / what is it for
        r"^\s*(what('?s| is| are)|tell me (about|more about)|explain|vad är|berätta om)\s+(an? |the |en |ett )?[\w\-åäöÅÄÖ]+(\s+[\w\-åäöÅÄÖ]+)?\s*\??\s*$",
    ],
    "side_effects": [
        r"\bside ?effects?\b", r"\badverse (effects?|reactions?)\b",
        r"\bbiverkning(ar)?\b",
    ],
    "pregnancy": [
        r"\bpregnan(t|cy)\b", r"\bbreast ?feed(ing)?\b", r"\bnursing\b",
        r"\bgravid(itet)?\b", r"\bamning\b", r"\bammar\b",
    ],
    "driving": [
        # Standalone "drive"/"driving" is unambiguous in a medication context
        # (unlike e.g. "take", which needs surrounding context) -- no need to
        # require a nearby word, and no need to worry about word order.
        r"\bdriv(e|ing)\b", r"\boperate machin", r"\bkör(a|ning)?\b",
        r"\bköra (bil|fordon)\b", r"\btrafik\b",
    ],
    "contraindication": [
        r"\bshould(n'?t| not) (i |you |take )", r"\bwho should ?n'?t\b",
        r"\bprecautions?\b", r"\bkontraindikation", r"\bförsiktighet\b",
        r"\bvem (bör|ska) inte\b", r"\bcontraindicat", r"\bwho (can|cannot|can't) (not )?(take|use)\b",
    ],
    "interaction": [
        r"\btake .* (together|with)\b", r"\bcombin(e|ing|ation)\b",
        r"\binteract(ion|s)?\b", r"\bmix(ing)?\b",
        r"\bta .* (tillsammans|samtidigt)\b", r"\bblanda\b", r"\bkombinera\b",
    ],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in _PATTERNS.items()}


def detect_intents(text: str, n_drugs: int, topics_found: list) -> list:
    """n_drugs: how many drugs were matched this turn (incl. carried-over).
    topics_found: output of topics.detect_topics(text).
    Always returns at least one intent -- falls back to a sensible default
    based on how many drugs are in play, so the pipeline always has somewhere
    to route evidence-gathering."""
    text = text or ""
    found = [key for key, patterns in _COMPILED.items() if any(p.search(text) for p in patterns)]

    if topics_found and "topic" not in found:
        found.append("topic")

    # "interaction" needs 2+ drugs to mean anything; with only 1, drop it and
    # let the fallback below pick a single-drug-appropriate intent instead.
    if "interaction" in found and n_drugs < 2:
        found.remove("interaction")

    if found:
        return found

    # No specific question detected -- default based on context rather than
    # silently returning nothing. Two+ drugs with no stated question almost
    # always means "do these interact"; one drug gets a general overview.
    return ["interaction"] if n_drugs >= 2 else ["overview"]


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "can I take this with alcohol?"
    print(detect_intents(text, n_drugs=1, topics_found=["alcohol"] if "alcohol" in text else []))