"""
doc_type_detector.py

Classifies a medical fax into a document type using weighted keyword scoring.

How it works
────────────
Each doc type has three tiers of keywords:
  strong  → 3 points each  (almost uniquely identify the doc type)
  medium  → 2 points each  (common to this type but not exclusive)
  weak    → 1 point each   (generic medical terms)

We lowercase the text, scan for each keyword, sum the points per type.
The type with the highest score wins.

Ambiguity detection
───────────────────
If the top two scores are within 20% of each other, we flag the result
as ambiguous and reduce confidence accordingly. The AI validation step
later can handle ambiguous cases more intelligently.

Confidence scoring for doc_type
────────────────────────────────
  confidence = top_score / max_possible_score_for_that_type

  This gives a relative measure: how many of the known signals for this
  type were actually present in the document.

Reference for medical fax vocabulary:
  CMS-1500 claim form field labels:
  https://www.cms.gov/medicare/cms-forms/cms-forms/downloads/cms1500.pdf
"""

import re
from state import DocType


# ── Keyword tiers per document type ───────────────────────────────────────────
# Each entry: { "strong": [...], "medium": [...], "weak": [...] }

DOC_TYPE_KEYWORDS: dict[DocType, dict] = {

    DocType.REFERRAL: {
        "strong": [
            "referring physician",
            "reason for referral",
            "referred to",
            "please see this patient",
            "referral letter",
        ],
        "medium": [
            "diagnosis",
            "specialist",
            "consult",
            "please evaluate",
            "clinical information",
        ],
        "weak": [
            "patient",
            "appointment",
            "history",
            "symptoms",
            "physician",
        ],
    },

    DocType.LAB_RESULT: {
        "strong": [
            "lab result",
            "laboratory result",
            "normal range",
            "reference range",
            "specimen collected",
            "specimen id",
            "resulted",
        ],
        "medium": [
            "panel",
            "collected",
            "units",
            "flag",
            "h",   # "H" marker for high values — common in lab printouts
            "l",   # "L" marker for low values
        ],
        "weak": [
            "test",
            "value",
            "result",
            "glucose",
            "hemoglobin",
        ],
    },

    DocType.PRESCRIPTION: {
        "strong": [
            "rx",
            "dispense",
            "refills",
            "sig:",
            "prescribe",
            "dea number",
            "days supply",
        ],
        "medium": [
            "dosage",
            "quantity",
            "take",
            "tablet",
            "capsule",
            "mg",
        ],
        "weak": [
            "medication",
            "pharmacy",
            "drug",
            "dose",
            "patient",
        ],
    },

    DocType.INSURANCE_AUTH: {
        "strong": [
            "prior authorization",
            "prior auth",
            "auth number",
            "approved units",
            "authorization number",
            "payer id",
        ],
        "medium": [
            "member id",
            "group number",
            "insurance",
            "coverage",
            "benefit",
        ],
        "weak": [
            "plan",
            "payer",
            "claim",
            "policy",
            "patient",
        ],
    },
}

# Points per tier
TIER_WEIGHTS = {"strong": 3, "medium": 2, "weak": 1}

# Ambiguity threshold — if runner-up score / top score > this, flag as ambiguous
AMBIGUITY_RATIO = 0.80


def detect_doc_type(text: str) -> tuple[DocType, float, dict]:
    """
    Classify the document type from plain text.

    Returns
    -------
    doc_type        : DocType enum value
    confidence      : float 0.0–1.0
    scores_detail   : dict with per-type scores (useful for debugging)
    """
    text_lower = text.lower()

    scores: dict[DocType, int] = {}
    matched_keywords: dict[DocType, list] = {}

    for doc_type, tiers in DOC_TYPE_KEYWORDS.items():
        total   = 0
        matched = []

        for tier_name, keywords in tiers.items():
            weight = TIER_WEIGHTS[tier_name]
            for kw in keywords:
                # Use word boundary matching to avoid substring false positives
                # e.g. "rx" should not match "proxy"
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, text_lower):
                    total += weight
                    matched.append((kw, tier_name, weight))

        scores[doc_type]          = total
        matched_keywords[doc_type] = matched

    # Find top and runner-up
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type,     top_score    = sorted_types[0]
    second_type,  second_score = sorted_types[1] if len(sorted_types) > 1 else (None, 0)

    # Handle zero scores — nothing matched at all
    if top_score == 0:
        return DocType.UNKNOWN, 0.0, {"scores": scores, "matched": matched_keywords}

    # Compute max possible score for this doc type
    # (if every keyword in every tier matched)
    max_possible = sum(
        TIER_WEIGHTS[tier] * len(kws)
        for tier, kws in DOC_TYPE_KEYWORDS[top_type].items()
    )
    raw_confidence = top_score / max_possible

    # Reduce confidence if result is ambiguous
    is_ambiguous = (second_score > 0) and (second_score / top_score >= AMBIGUITY_RATIO)
    if is_ambiguous:
        raw_confidence *= 0.75  # penalise — AI should double-check
        print(f"  ⚠️  Ambiguous: {top_type} ({top_score}pt) vs {second_type} ({second_score}pt)")
    else:
        print(f"  ✅ Doc type: {top_type} (score {top_score}/{max_possible}, "
              f"confidence {raw_confidence:.2f})")

    scores_detail = {
        "scores":      {k.value: v for k, v in scores.items()},
        "matched":     {k.value: v for k, v in matched_keywords.items()},
        "is_ambiguous": is_ambiguous,
    }

    return top_type, round(raw_confidence, 3), scores_detail