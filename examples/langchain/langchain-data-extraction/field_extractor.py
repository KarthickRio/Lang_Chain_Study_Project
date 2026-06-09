"""
field_extractor.py

Extracts structured fields from plain text using regex patterns.
Which fields to look for depends on the doc_type detected earlier.

Design principle: extract what you can, flag what you can't.
──────────────────────────────────────────────────────────────
We never guess or hallucinate a value. If a field isn't found,
it goes into missing_fields. The AI validation step is responsible
for deciding whether that's a real problem or just a sparse fax.

Confidence scoring
──────────────────
  confidence = fields_found / total_expected_fields_for_this_doc_type

This gives the AI a signal: a 0.4 confidence referral is worth
sending to the AI; a 0.9 confidence lab result may go straight
to synthesis.

Date normalisation
──────────────────
Medical faxes use wildly inconsistent date formats:
  "01/12/1980"  "Jan 12, 1980"  "12-JAN-80"  "2024.01.12"
We use python-dateutil to normalise all of them to ISO 8601:
  "1980-01-12"

Install: pip install python-dateutil
Docs: https://dateutil.readthedocs.io/en/stable/
"""

import re
from dateutil import parser as date_parser
from state import DocType


# ── Expected fields per doc type ──────────────────────────────────────────────
# Used to compute confidence: fields_found / len(EXPECTED_FIELDS[doc_type])

EXPECTED_FIELDS: dict[DocType, list] = {
    DocType.REFERRAL: [
        "patient_name", "dob", "referring_physician",
        "referring_npi", "specialist_name", "diagnosis",
        "icd_code", "urgency", "date", "insurance_id",
    ],
    DocType.LAB_RESULT: [
        "patient_name", "dob", "ordering_physician",
        "collection_date", "result_date", "accession_number",
    ],
    DocType.PRESCRIPTION: [
        "patient_name", "dob", "prescriber",
        "npi", "drug_name", "dosage",
        "quantity", "refills", "date",
    ],
    DocType.INSURANCE_AUTH: [
        "patient_name", "member_id", "group_number",
        "auth_number", "payer_name", "date",
    ],
}


# ── Regex patterns ─────────────────────────────────────────────────────────────
# Each pattern tries several common label variations.
# re.IGNORECASE is always set — fax formatting is inconsistent.
#
# Pattern structure:
#   label_variation_1|label_variation_2   followed by   [\s:]+   then   capture group
#
# The capture group uses a non-greedy match that stops at:
#   - newline  \n
#   - tab      \t
#   - slash    /   (date boundaries)
#   - or a fixed length (for NPI, ICD codes etc.)

PATTERNS: dict[str, str] = {

    # ── Universal (used across all doc types) ─────────────────────────
    "patient_name": r'(?:patient(?:\s+name)?|pt\.?\s+name)[:\s]+([A-Za-z,\s\-\.]+?)(?:\n|DOB|Date of Birth|$)',
    "dob":          r'(?:dob|date of birth|birth(?:date)?)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})',
    "date":         r'(?:date|fax date|date of fax)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    "insurance_id": r'(?:insurance(?:\s+id)?|member(?:\s+id)?)[:\s]+([A-Za-z0-9\-]+)',

    # ── Referral ──────────────────────────────────────────────────────
    "referring_physician": r'(?:referring(?:\s+physician)?|referred\s+by)[:\s]+(?:Dr\.?\s+)?([A-Za-z,\s\-\.]+?)(?:\n|NPI|$)',
    "referring_npi":       r'(?:npi(?:\s+number)?|referring\s+npi)[:\s]+(\d{10})',
    "specialist_name":     r'(?:referred\s+to|specialist|consult(?:ant)?)[:\s]+(?:Dr\.?\s+)?([A-Za-z,\s\-\.]+?)(?:\n|$)',
    "diagnosis":           r'(?:diagnosis|dx|clinical\s+diagnosis)[:\s]+([^\n]+)',
    "icd_code":            r'(?:icd(?:\-?10)?(?:\s+code)?)[:\s]+([A-Z]\d{2}(?:\.\d{1,4})?)',
    "urgency":             r'(?:urgency|priority)[:\s]+(routine|urgent|emergent|stat)',

    # ── Lab result ────────────────────────────────────────────────────
    "ordering_physician":  r'(?:ordering(?:\s+physician)?|ordered\s+by)[:\s]+(?:Dr\.?\s+)?([A-Za-z,\s\-\.]+?)(?:\n|$)',
    "collection_date":     r'(?:collected?(?:\s+date)?|collection\s+date)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    "result_date":         r'(?:resulted?(?:\s+date)?|report\s+date)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    "accession_number":    r'(?:accession(?:\s+(?:number|no|#))?)[:\s]+([A-Za-z0-9\-]+)',

    # ── Prescription ─────────────────────────────────────────────────
    "prescriber":  r'(?:prescriber|prescribing\s+physician|prescrib(?:ed)?\s+by)[:\s]+(?:Dr\.?\s+)?([A-Za-z,\s\-\.]+?)(?:\n|NPI|$)',
    "npi":         r'(?<!\breferring\s)(?:npi(?:\s+number)?)[:\s]+(\d{10})',
    "drug_name":   r'(?:drug|medication|rx\s+name|product)[:\s]+([A-Za-z0-9\s\-]+?)(?:\n|\d+\s*mg|$)',
    "dosage":      r'(\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|units?))',
    "quantity":    r'(?:qty|quantity|#)[:\s]+(\d+)',
    "refills":     r'(?:refills?)[:\s]+(\d+|none|no\s+refills?)',

    # ── Insurance auth ────────────────────────────────────────────────
    "member_id":    r'(?:member(?:\s+id)?|member(?:\s+number)?)[:\s]+([A-Za-z0-9\-]+)',
    "group_number": r'(?:group(?:\s+(?:number|no|#))?)[:\s]+([A-Za-z0-9\-]+)',
    "auth_number":  r'(?:auth(?:orization)?(?:\s+(?:number|no|#))?)[:\s]+([A-Za-z0-9\-]+)',
    "payer_name":   r'(?:payer|insurance\s+company|plan\s+name)[:\s]+([A-Za-z\s\-]+?)(?:\n|$)',
}


def _normalise_date(raw: str) -> str:
    """
    Convert any date string to ISO 8601 (YYYY-MM-DD).
    Returns the original string if parsing fails.
    """
    try:
        parsed = date_parser.parse(raw, dayfirst=False)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return raw.strip()


def extract_fields(text: str, doc_type: DocType) -> tuple[dict, list, float]:
    """
    Extract known fields for the given doc_type from plain text.

    Returns
    -------
    extracted_fields : dict  — field_name → value (cleaned string)
    missing_fields   : list  — field names that were not found
    confidence       : float — fields_found / total_expected
    """
    expected      = EXPECTED_FIELDS.get(doc_type, [])
    extracted     = {}
    date_fields   = {"dob", "date", "collection_date", "result_date"}

    for field in expected:
        pattern = PATTERNS.get(field)
        if not pattern:
            continue  # no pattern defined yet — treated as missing

        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw_value = match.group(1).strip()

            # Normalise dates to ISO 8601
            if field in date_fields:
                raw_value = _normalise_date(raw_value)

            extracted[field] = raw_value

    missing = [f for f in expected if f not in extracted]

    confidence = len(extracted) / len(expected) if expected else 0.0

    print(f"  Fields found:   {list(extracted.keys())}")
    if missing:
        print(f"  Fields missing: {missing}")
    print(f"  Confidence:     {confidence:.2f}")

    return extracted, missing, round(confidence, 3)