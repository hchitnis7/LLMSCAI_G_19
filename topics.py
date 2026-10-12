"""Detects when the user is asking about a medicine alongside something that
ISN'T another medicine -- alcohol, grapefruit, driving, pregnancy, caffeine,
smoking. These route to Knowledge.topic_evidence() instead of pair_evidence(),
since there's no second drug to look up by ATC code.

English + Swedish keyword lists, kept deliberately simple (word-boundary
matching) in the same style as matcher.py's stopwords and prompts.py's
language detection -- fast, deterministic, no extra model call needed.
"""
import re

# topic key -> (terms to search FASS text for for this topic, trigger phrases
# in the user's own message that indicate they're asking about it)
TOPICS = {
    "alcohol": {
        "search_terms": ["alkohol", "alcohol", "sprit", "vin", "öl", "beer", "wine"],
        "triggers": ["alcohol", "alkohol", "beer", "öl", "wine", "vin", "sprit",
                    "drink", "dricka", "drunk", "full"],
    },
    "grapefruit": {
        "search_terms": ["grapefrukt", "grapefruit"],
        "triggers": ["grapefruit", "grapefrukt"],
    },
    "caffeine": {
        "search_terms": ["koffein", "caffeine", "kaffe", "coffee"],
        "triggers": ["caffeine", "koffein", "coffee", "kaffe", "energy drink", "energidryck"],
    },
    "st_johns_wort": {
        "search_terms": ["johannesört", "st. johns wort", "st johns wort", "johannesort"],
        "triggers": ["st john's wort", "st johns wort", "johannesört", "johannesort"],
    },
    "smoking": {
        "search_terms": ["rökning", "tobak", "smoking", "nikotin", "nicotine"],
        "triggers": ["smoking", "smoke", "smokes", "smoker", "rökning", "roker", "röker", "tobacco", "tobak", "cigarett"],
    },
    # NOTE: driving is intentionally NOT a topic here, even though it's a
    # "what can I combine this with" style question. Unlike alcohol/
    # grapefruit/caffeine (which have no dedicated FASS section and only show
    # up scattered inside interaction/warning text), driving has its own
    # numbered FASS section (4.7). It's handled as an INTENT in intents.py,
    # routed to Knowledge.section_evidence(drug, "driving", ...) instead of
    # topic_evidence() -- keeping it here too would double-route it.
}

_WORD_RE = re.compile(r"\b[\wåäöÅÄÖ'-]+\b")


def detect_topics(text: str) -> list:
    """Returns a list of topic keys (from TOPICS) mentioned in `text`.
    Checked against whole trigger phrases (which may contain spaces), not
    single tokens, so multi-word triggers like 'st john's wort' work."""
    low = (text or "").lower()
    found = []
    for key, spec in TOPICS.items():
        if any(trigger in low for trigger in spec["triggers"]):
            found.append(key)
    return found


def search_terms_for(topic_keys) -> list:
    """Flattened FASS-text search terms for a list of topic keys, for
    Knowledge.topic_evidence()."""
    terms = []
    for k in topic_keys:
        terms += TOPICS.get(k, {}).get("search_terms", [])
    return terms


def topic_display_names(topic_keys) -> list:
    """Human-readable names for the detected topics, for logging/prompts."""
    names = {"alcohol": "alcohol", "grapefruit": "grapefruit", "caffeine": "caffeine",
            "st_johns_wort": "St John's wort", "smoking": "smoking/tobacco", "driving": "driving"}
    return [names.get(k, k) for k in topic_keys]


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "can I drink alcohol with this?"
    found = detect_topics(text)
    print("Topics found:", found)
    print("Search terms:", search_terms_for(found))