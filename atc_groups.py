"""ATC (WHO) therapeutic groups, used as a data-backed 'what kind of medicine is this'
fallback when no FASS section 4.1 (indications) has been collected for a drug.

The ATC code itself comes from drugs_lookup.csv, so this is derived from the project's
own data -- it never invents a specific use. Level 1 = anatomical main group (first
letter), level 2 = therapeutic subgroup (first three characters).
"""

ATC_LEVEL1 = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood-forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals (skin)",
    "G": "Genito-urinary system and sex hormones",
    "H": "Systemic hormonal preparations (excluding sex hormones and insulins)",
    "J": "Anti-infectives for systemic use",
    "L": "Antineoplastic and immunomodulating agents",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products, insecticides and repellents",
    "R": "Respiratory system",
    "S": "Sensory organs (eyes and ears)",
    "V": "Various",
}

ATC_LEVEL2 = {
    "A01": "Stomatological (mouth and teeth) preparations", "A02": "Drugs for acid-related disorders",
    "A03": "Drugs for functional gastrointestinal disorders", "A04": "Antiemetics and antinauseants",
    "A05": "Bile and liver therapy", "A06": "Drugs for constipation",
    "A07": "Antidiarrheals and intestinal anti-inflammatory/anti-infective agents",
    "A08": "Antiobesity preparations", "A09": "Digestives, including enzymes",
    "A10": "Drugs used in diabetes", "A11": "Vitamins", "A12": "Mineral supplements",
    "A16": "Other alimentary tract and metabolism products",
    "B01": "Antithrombotic agents (blood thinners)", "B02": "Antihemorrhagics",
    "B03": "Antianemic preparations", "B05": "Blood substitutes and perfusion solutions",
    "C01": "Cardiac therapy", "C02": "Antihypertensives", "C03": "Diuretics",
    "C04": "Peripheral vasodilators", "C05": "Vasoprotectives", "C07": "Beta blocking agents",
    "C08": "Calcium channel blockers", "C09": "Agents acting on the renin-angiotensin system",
    "C10": "Lipid-modifying agents",
    "D01": "Antifungals for skin use", "D04": "Antipruritics (anti-itch)", "D05": "Antipsoriatics",
    "D06": "Antibiotics and chemotherapeutics for skin use", "D07": "Corticosteroids for skin use",
    "D10": "Anti-acne preparations",
    "G01": "Gynecological anti-infectives and antiseptics", "G02": "Other gynecologicals",
    "G03": "Sex hormones and modulators of the genital system", "G04": "Urologicals",
    "H01": "Pituitary and hypothalamic hormones", "H02": "Corticosteroids for systemic use",
    "H03": "Thyroid therapy", "H04": "Pancreatic hormones", "H05": "Calcium homeostasis",
    "J01": "Antibacterials for systemic use (antibiotics)", "J02": "Antimycotics for systemic use (antifungals)",
    "J04": "Antimycobacterials", "J05": "Antivirals for systemic use", "J06": "Immune sera and immunoglobulins",
    "J07": "Vaccines",
    "L01": "Antineoplastic agents (cancer treatment)", "L02": "Endocrine therapy (cancer)",
    "L03": "Immunostimulants", "L04": "Immunosuppressants",
    "M01": "Anti-inflammatory and antirheumatic products", "M02": "Topical products for joint and muscular pain",
    "M03": "Muscle relaxants", "M04": "Antigout preparations", "M05": "Drugs for treatment of bone diseases",
    "N01": "Anesthetics", "N02": "Analgesics (pain relief)", "N03": "Antiepileptics",
    "N04": "Anti-Parkinson drugs", "N05": "Psycholeptics (antipsychotics, anxiolytics, sleep medicines)",
    "N06": "Psychoanaleptics (antidepressants, ADHD and dementia medicines)",
    "N07": "Other nervous system drugs",
    "P01": "Antiprotozoals", "P02": "Anthelmintics", "P03": "Ectoparasiticides (e.g. for lice or scabies)",
    "R01": "Nasal preparations", "R02": "Throat preparations", "R03": "Drugs for obstructive airway diseases",
    "R05": "Cough and cold preparations", "R06": "Antihistamines for systemic use",
    "S01": "Ophthalmologicals (eye)", "S02": "Otologicals (ear)",
}


def atc_group_text(atc: str) -> str:
    """Human-readable group description for an ATC code, or '' if unknown."""
    atc = (atc or "").strip().upper()
    if not atc:
        return ""
    lvl2 = ATC_LEVEL2.get(atc[:3])
    lvl1 = ATC_LEVEL1.get(atc[:1])
    if lvl2 and lvl1:
        return f"ATC {atc[:3]}: {lvl2} (main group {atc[0]}: {lvl1})"
    if lvl1:
        return f"ATC main group {atc[0]}: {lvl1}"
    return ""
