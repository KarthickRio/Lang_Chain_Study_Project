"""
doc_type_detector.py

Classifies a US medical fax into a document type using weighted keyword scoring.

How scoring works
──────────────────
strong  → 3 points  (vocabulary almost unique to this doc type)
medium  → 2 points  (common to this type, rarely in others)
weak    → 1 point   (generic medical terms, appear everywhere)

Winner = highest total score.
Confidence = winner_score / max_possible_score_for_that_type.

Ambiguity rule
───────────────
If runner-up score / winner score >= 0.80 → flag as ambiguous,
reduce confidence by 25%. AI validation handles ambiguous cases.

Changes from previous version
───────────────────────────────
Added RX_RENEWAL doc type with keywords from real US renewal fax:
  "rxrenewal request", "erx id", "ncpdp", "last filled",
  "drug prescribed", "qty. prescribed", "qty. dispensed"
These are distinct from PRESCRIPTION keywords which cover
newly written Rx forms rather than pharmacy renewal requests.
"""

import re
from state import DocType


DOC_TYPE_KEYWORDS: dict[DocType, dict] = {

    # ── RX_RENEWAL ────────────────────────────────────────────────────
    # Keywords taken directly from real RxRenewal fax text.
    # "ncpdp" and "erx id" are unique identifiers that only appear
    # on electronic prescription renewal requests — high signal.
    # "last filled" only appears on renewals, not new prescriptions.
    DocType.RX_RENEWAL: {
        "strong": [
            "rxrenewal request",
            "rx renewal request",
            "erx id",           # electronic Rx transaction ID
            "ncpdp",            # National Council for Prescription Drug Programs
            "last filled",      # renewal-specific — when drug was last dispensed
            "drug prescribed",  # exact label on renewal fax forms
            "qty. prescribed",  # exact label with period — renewal forms use this
            "qty. dispensed",   # only on renewals — what pharmacy actually gave
        ],
        "medium": [
            "date written",
            "dispense as written",
            "rx number",
            "drug dispensed",
            "sig",
            "refills",
            "substitution permitted",
        ],
        "weak": [
            "pharmacy",
            "prescriber",
            "patient",
            "dea",
            "npi",
        ],
    },

    # ── PRESCRIPTION ──────────────────────────────────────────────────
    # Standard newly-written prescription form.
    # Distinct from RX_RENEWAL — no "last filled", no "ncpdp",
    # no "qty. dispensed". Uses "quantity" not "qty. prescribed".
    DocType.PRESCRIPTION: {
        "strong": [
            "rx",
            "dispense",
            "refills",
            "sig:",
            "dea number",
            "days supply",
            "prescribe",
        ],
        "medium": [
            "dosage",
            "quantity",
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

    # ── REFERRAL ──────────────────────────────────────────────────────
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

    # ── LAB_RESULT ────────────────────────────────────────────────────
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
        ],
        "weak": [
            "test",
            "value",
            "result",
            "glucose",
            "hemoglobin",
        ],
    },

    # ── INSURANCE_AUTH ────────────────────────────────────────────────
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

TIER_WEIGHTS   = {"strong": 3, "medium": 2, "weak": 1}
AMBIGUITY_RATIO = 0.80


def detect_doc_type(text: str) -> tuple[DocType, float, dict]:
    """
    Classify document type from plain text.

    Returns
    -------
    doc_type      : DocType enum
    confidence    : float 0.0–1.0
    scores_detail : dict  (for debugging)
    """
    text_lower = text.lower()
    scores:           dict[DocType, int]  = {}
    matched_keywords: dict[DocType, list] = {}

    for doc_type, tiers in DOC_TYPE_KEYWORDS.items():
        total   = 0
        matched = []
        for tier_name, keywords in tiers.items():
            weight = TIER_WEIGHTS[tier_name]
            for kw in keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, text_lower):
                    total += weight
                    matched.append((kw, tier_name, weight))
        scores[doc_type]           = total
        matched_keywords[doc_type] = matched

    sorted_types  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type,    top_score    = sorted_types[0]
    second_type, second_score = sorted_types[1] if len(sorted_types) > 1 else (None, 0)

    if top_score == 0:
        return DocType.UNKNOWN, 0.0, {"scores": scores, "matched": matched_keywords}

    max_possible = sum(
        TIER_WEIGHTS[tier] * len(kws)
        for tier, kws in DOC_TYPE_KEYWORDS[top_type].items()
    )
    raw_confidence = top_score / max_possible

    is_ambiguous = (second_score > 0) and (second_score / top_score >= AMBIGUITY_RATIO)
    if is_ambiguous:
        raw_confidence *= 0.75
        print(f"  ⚠️  Ambiguous: {top_type} ({top_score}pt) vs {second_type} ({second_score}pt)")
    else:
        print(f"  ✅ Doc type: {top_type} "
              f"(score {top_score}/{max_possible}, confidence {raw_confidence:.2f})")

    return top_type, round(raw_confidence, 3), {
        "scores":       {k.value: v for k, v in scores.items()},
        "matched":      {k.value: v for k, v in matched_keywords.items()},
        "is_ambiguous": is_ambiguous,
    }