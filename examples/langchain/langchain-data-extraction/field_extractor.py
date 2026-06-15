"""
field_extractor.py

Extracts structured fields from plain text using regex patterns.
Which fields to extract depends on doc_type detected by doc_type_detector.

Changes from previous version
───────────────────────────────
1. Field names updated to US standard terminology:
     prescriber   → prescriber_name
     drug_name    → drug_prescribed
     dosage       → strength
     npi          → prescriber_npi (separate from pharmacy_npi)

2. Added RX_RENEWAL expected fields and patterns — written against
   real RxRenewal fax labels (Drug Prescribed:, Qty. Prescribed:,
   Dob:, SIG, Last Filled:, NCPDP, eRx ID)

3. Two NPI patterns — pharmacy NPI (near NCPDP) vs prescriber NPI
   (near DEA). Real renewal faxes have both on the same document.

4. OCR date cleanup — Option B:
   If year has 5+ digits (OCR error like "20125"), mark as None.
   Goes into missing_fields. AI tool handles it.
   We do NOT try to fix it here — that is the AI's job.

5. phi_fields tracking — returns list of field names that contain PHI
   so state can record which fields need HIPAA-aware handling.

Confidence scoring
──────────────────
  confidence = fields_found / total_expected_for_this_doc_type
"""

import re
from dateutil import parser as date_parser
from state import DocType


# ── PHI field names ───────────────────────────────────────────────────────────
# These field names contain Protected Health Information.
# Returned separately so state can track them for HIPAA awareness.

PHI_FIELD_NAMES = {
    "patient_name", "dob", "patient_address",
    "patient_phone", "rx_number", "erx_id"
}


# ── Expected fields per doc type ──────────────────────────────────────────────
# Confidence = fields_found / len(EXPECTED_FIELDS[doc_type])

EXPECTED_FIELDS: dict[DocType, list] = {

    # ── RX_RENEWAL — based on real fax structure ───────────────────────
    # Pharmacy block: pharmacy_name, pharmacy_npi, ncpdp
    # Prescriber block: prescriber_name, prescriber_npi, dea_number
    # Patient block: patient_name, dob
    # Drug block: drug_prescribed, strength, qty_prescribed,
    #             date_written, last_filled, sig, refills, rx_number
    DocType.RX_RENEWAL: [
        "patient_name",
        "dob",
        "prescriber_name",
        "prescriber_npi",
        "dea_number",
        "drug_prescribed",
        "strength",
        "qty_prescribed",
        "date_written",
        "last_filled",
        "sig",
        "refills",
        "rx_number",
        "pharmacy_name",
        "pharmacy_npi",
    ],

    DocType.PRESCRIPTION: [
        "patient_name",
        "dob",
        "prescriber_name",
        "prescriber_npi",
        "dea_number",
        "drug_prescribed",
        "strength",
        "qty_prescribed",
        "date_written",
        "sig",
        "refills",
    ],

    DocType.REFERRAL: [
        "patient_name", "dob", "prescriber_name",
        "prescriber_npi", "specialist_name", "diagnosis",
        "icd_code", "urgency", "date_written", "insurance_id",
    ],

    DocType.LAB_RESULT: [
        "patient_name", "dob", "ordering_physician",
        "collection_date", "result_date", "accession_number",
    ],

    DocType.INSURANCE_AUTH: [
        "patient_name", "member_id", "group_number",
        "auth_number", "payer_name", "date_written",
    ],
}


# ── Regex patterns ─────────────────────────────────────────────────────────────
# Written against real US medical fax label vocabulary.
# All patterns use re.IGNORECASE.
#
# Two NPI patterns explained:
#   pharmacy_npi  → appears right after NCPDP line in renewal faxes
#   prescriber_npi → appears after DEA line OR in prescriber block
#   Using context (what comes before) to distinguish them.

PATTERNS: dict[str, str] = {

    # ── Patient fields ─────────────────────────────────────────────────
    # Real label: "Patient: Caitlyn Davis" or "Patient Name: John Doe"
    "patient_name": r'Patient(?:\s+Name)?:\s+([A-Za-z\s\-\.]+?)(?:\n|Address|DOB|Dob|$)',

    # Real label: "Dob: 02/08/2007" or "Date of Birth: 05/18/1988"
    # Note: renewal faxes use "Dob:" (capital D lowercase ob)
    "dob": r'(?:Dob|DOB|Date\s+of\s+Birth)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',

    # ── Prescriber fields ──────────────────────────────────────────────
    # Real label: "Prescriber: AMANDA MARIE WEINDL"
    # Stops at newline, Address, NPI, or DEA
    "prescriber_name": r'Prescriber:\s+([A-Za-z\s\.\,]+?)(?:\n|Address|NPI|DEA|$)',

    # Prescriber NPI — appears AFTER DEA line in the prescriber block
    # Pattern: look for NPI that follows DEA on a nearby line
    # Real sequence: "DEA: FW6804935\nNPI: 1699091280"
    "prescriber_npi": r'DEA:.*?\n.*?NPI:\s*(\d{10})',

    # Pharmacy NPI — appears right after NCPDP line
    # Real sequence: "NCPDP: 4927313\nNPI: 1932155447"
    "pharmacy_npi": r'NCPDP.*?\n.*?NPI:\s*(\d{10})',

    # Real label: "DEA: FW6804935"
    # US DEA format: 2 letters + 7 digits
    "dea_number": r'DEA:\s+([A-Z]{2}\d{7})',

    # ── Drug fields ────────────────────────────────────────────────────
    # Real label: "Drug Prescribed: FLUoxetine HCI Oral Capsule 40 MG"
    # Stops at newline — drug name is always single line
    "drug_prescribed": r'Drug\s+(?:Prescribed|Dispensed):\s+([^\n]+?)(?:\s+\d+\s*MG|\n|$)',

    # Strength extracted from drug name line or standalone
    # Matches: "40 MG", "20 mg", "500 MCG", "10 ML"
    "strength": r'(\d+(?:\.\d+)?\s*(?:MG|MCG|ML|G|UNITS?))\b',

    # Real label: "Qty. Prescribed: 30 Thirty"
    # The period in "Qty." is escaped. Stops at space+word (written amount)
    "qty_prescribed": r'Qty\.?\s*(?:Prescribed|Dispensed)?:\s+(\d+)',

    # Real label: "Date Written: 09/09/2025"
    # OCR cleanup: if year part has 5+ digits → return None (Option B)
    "date_written": r'Date\s+Written:\s+(\d{1,2}\/\d{1,2}\/\d{4,5})',

    # Real label: "Last Filled: 09/10/2025" — only on renewal faxes
    "last_filled": r'Last\s+Filled:\s+(\d{1,2}\/\d{1,2}\/\d{4})',

    # Real label: "SIG Take Capsule by mouth once daily"
    # or "SIG: Take Capsule by mouth once daily"
    # Colon is optional — real faxes vary
    "sig": r'SIG:?\s+([^\n]+)',

    # Real label: "Refills" followed by blank or number
    # On renewal faxes refills is often blank → returns None → missing_fields
    "refills": r'Refills\s*[:\s]+(\d+)',

    # Real label: "Rx Number: 6785160"
    "rx_number": r'Rx\s*(?:Number|No\.?):\s+(\d+)',

    # Real label: "Pharmacy: Safeway Pharmacy #0258"
    "pharmacy_name": r'Pharmacy:\s+([^\n]+?)(?:\n|$)',

    # eRx ID — electronic prescription transaction identifier
    "erx_id": r'eRx\s*ID:\s*(\d+)',

    # ── Shared fields across doc types ─────────────────────────────────
    "insurance_id":       r'(?:insurance(?:\s+id)?|member(?:\s+id)?)[:\s]+([A-Za-z0-9\-]+)',
    "icd_code":           r'(?:icd(?:\-?10)?(?:\s+code)?)[:\s]+([A-Z]\d{2}(?:\.\d{1,4})?)',
    "diagnosis":          r'(?:diagnosis|dx|clinical\s+diagnosis)[:\s]+([^\n]+)',
    "urgency":            r'(?:urgency|priority)[:\s]+(routine|urgent|emergent|stat)',
    "ordering_physician": r'(?:ordering(?:\s+physician)?|ordered\s+by)[:\s]+([A-Za-z,\s\-\.]+?)(?:\n|$)',
    "collection_date":    r'(?:collected?(?:\s+date)?)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    "result_date":        r'(?:resulted?(?:\s+date)?|report\s+date)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    "accession_number":   r'(?:accession(?:\s+(?:number|no|#))?)[:\s]+([A-Za-z0-9\-]+)',
    "member_id":          r'(?:member(?:\s+id)?|member(?:\s+number)?)[:\s]+([A-Za-z0-9\-]+)',
    "group_number":       r'(?:group(?:\s+(?:number|no|#))?)[:\s]+([A-Za-z0-9\-]+)',
    "auth_number":        r'(?:auth(?:orization)?(?:\s+(?:number|no|#))?)[:\s]+([A-Za-z0-9\-]+)',
    "payer_name":         r'(?:payer|insurance\s+company|plan\s+name)[:\s]+([A-Za-z\s\-]+?)(?:\n|$)',
    "specialist_name":    r'(?:referred\s+to|specialist|consult(?:ant)?)[:\s]+([A-Za-z,\s\-\.]+?)(?:\n|$)',
}

DATE_FIELDS = {"dob", "date_written", "last_filled", "collection_date", "result_date"}


def _normalise_date(raw: str, field_name: str) -> str | None:
    """
    Normalise date string to ISO 8601 (YYYY-MM-DD).

    Option B: if year part has 5+ digits (OCR error like "20125"),
    return None so field goes into missing_fields.
    AI validation tool handles it from there.
    """
    # Extract year part — last segment after / or -
    parts = re.split(r'[\/\-\.]', raw.strip())
    if parts:
        year_part = parts[-1]
        if len(year_part) > 4:
            print(f"  ⚠️  {field_name}: OCR year error '{raw}' → marked missing")
            return None

    try:
        parsed = date_parser.parse(raw, dayfirst=False)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return raw.strip()


def extract_fields(
    text: str,
    doc_type: DocType
) -> tuple[dict, list, float, list]:
    """
    Extract known fields for doc_type from plain text.

    Returns
    -------
    extracted_fields : dict  — field_name → value
    missing_fields   : list  — expected fields not found
    confidence       : float — fields_found / total_expected
    phi_fields_found : list  — which extracted fields contain PHI
    """
    expected         = EXPECTED_FIELDS.get(doc_type, [])
    extracted        = {}
    phi_fields_found = []

    for field in expected:
        pattern = PATTERNS.get(field)
        if not pattern:
            continue

        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            raw_value = match.group(1).strip()

            # Date normalisation with OCR year check
            if field in DATE_FIELDS:
                normalised = _normalise_date(raw_value, field)
                if normalised is None:
                    # OCR error — treat as missing, AI handles it
                    continue
                raw_value = normalised

            # Skip empty captures
            if not raw_value:
                continue

            extracted[field] = raw_value

            # Track PHI fields
            if field in PHI_FIELD_NAMES:
                phi_fields_found.append(field)

    missing    = [f for f in expected if f not in extracted]
    confidence = len(extracted) / len(expected) if expected else 0.0

    print(f"  Fields found:   {list(extracted.keys())}")
    if missing:
        print(f"  Fields missing: {missing}")
    print(f"  Confidence:     {confidence:.2f}")

    return extracted, missing, round(confidence, 3), phi_fields_found