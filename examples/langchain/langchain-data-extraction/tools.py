"""
tools.py

LangChain tools for the AI extraction node.
The LLM calls these autonomously based on what it finds in the document.

IMPORTANT: The docstring of each @tool function is what the LLM reads
to decide when and how to call the tool. Write docstrings as instructions
to the LLM, not as developer documentation.

Three tools:
  1. verify_npi        — NPPES government API (free, no key)
  2. lookup_ndc        — openFDA drug API (free, no key)
  3. validate_field_value — local rules, no API

APIs used:
  NPPES:   https://npiregistry.cms.hhs.gov/api/?number={npi}&version=2.1
  openFDA: https://api.fda.gov/drug/ndc.json?search=generic_name:{drug}
"""

import re
import requests
from langchain_core.tools import tool


# ── Timeouts ───────────────────────────────────────────────────────────────────
# Government APIs can be slow — 10 seconds is safe for POC
API_TIMEOUT = 10


# ── Tool 1: verify_npi ────────────────────────────────────────────────────────

@tool
def verify_npi(npi_number: str) -> dict:
    """
    Verify a US National Provider Identifier (NPI) against the NPPES registry.

    Call this tool whenever you extract a prescriber_npi from the document.
    It confirms the NPI belongs to a real licensed US healthcare provider
    and returns their name and specialty so you can check it matches
    the prescriber name found in the document.

    Also call this if you are unsure whether an extracted NPI belongs to
    the prescriber or the pharmacy — the entity_type field will tell you:
    'Individual' means a person (prescriber), 'Organization' means a pharmacy.

    Args:
        npi_number: The 10-digit NPI number as a string. Example: "1699091280"

    Returns dict with:
        valid         : bool   — True if NPI exists in registry
        provider_name : str    — Full name from registry
        credential    : str    — e.g. "NP", "MD", "DO"
        specialty     : str    — Primary taxonomy description
        state         : str    — State of practice
        entity_type   : str    — "Individual" or "Organization"
        error         : str    — Present only if API call failed
    """
    print(f"\n  🔧 Tool: verify_npi({npi_number})")

    # Basic format check before hitting the API
    if not re.fullmatch(r'\d{10}', npi_number.strip()):
        result = {
            "valid":         False,
            "provider_name": "",
            "credential":    "",
            "specialty":     "",
            "state":         "",
            "entity_type":   "",
            "error":         f"Invalid format — NPI must be exactly 10 digits, got '{npi_number}'"
        }
        print(f"  ← Result: invalid format")
        return result

    try:
        url      = f"https://npiregistry.cms.hhs.gov/api/?number={npi_number}&version=2.1"
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data     = response.json()

        result_count = data.get("result_count", 0)

        if result_count == 0:
            result = {
                "valid":         False,
                "provider_name": "",
                "credential":    "",
                "specialty":     "",
                "state":         "",
                "entity_type":   "",
                "error":         f"NPI {npi_number} not found in NPPES registry"
            }
            print(f"  ← Result: not found in registry")
            return result

        # Parse first result
        provider    = data["results"][0]
        basic       = provider.get("basic", {})
        entity_type = provider.get("enumeration_type", "")

        # Name differs by entity type
        if entity_type == "NPI-1":   # Individual
            first      = basic.get("first_name", "")
            last       = basic.get("last_name",  "")
            credential = basic.get("credential", "")
            name       = f"{first} {last}".strip()
            entity_str = "Individual"
        else:                         # Organization (NPI-2)
            name       = basic.get("organization_name", "")
            credential = ""
            entity_str = "Organization"

        # Primary taxonomy = specialty
        taxonomies = provider.get("taxonomies", [])
        primary_tx = next((t for t in taxonomies if t.get("primary")), {})
        specialty  = primary_tx.get("desc", "")

        # State from practice address
        addresses  = provider.get("addresses", [])
        practice   = next((a for a in addresses
                           if a.get("address_purpose") == "LOCATION"), {})
        state      = practice.get("state", "")

        result = {
            "valid":         True,
            "provider_name": name,
            "credential":    credential,
            "specialty":     specialty,
            "state":         state,
            "entity_type":   entity_str,
            "error":         ""
        }
        print(f"  ← Result: {name} | {credential} | {specialty} | {state} | {entity_str}")
        return result

    except requests.Timeout:
        result = {
            "valid": False, "provider_name": "", "credential": "",
            "specialty": "", "state": "", "entity_type": "",
            "error": "NPPES API timeout"
        }
        print(f"  ← Result: API timeout")
        return result

    except Exception as e:
        result = {
            "valid": False, "provider_name": "", "credential": "",
            "specialty": "", "state": "", "entity_type": "",
            "error": str(e)
        }
        print(f"  ← Result: error — {e}")
        return result


# ── Tool 2: lookup_ndc ────────────────────────────────────────────────────────

@tool
def lookup_ndc(drug_name: str) -> dict:
    """
    Look up a drug in the FDA National Drug Code (NDC) directory.

    Call this tool whenever you extract a drug_prescribed field.
    It normalises inconsistent OCR drug names (like 'FLUoxetine HCI Oral Capsule')
    to the FDA standard name and returns the NDC code which uniquely
    identifies the drug for pharmacy dispensing systems.

    Also useful for confirming the extracted strength matches what is
    on file for this drug — a mismatch may indicate an OCR error.

    Args:
        drug_name: The drug name string from the document.
                   Can be brand name, generic name, or partial name.
                   Example: "FLUoxetine HCI Oral Capsule"

    Returns dict with:
        found          : bool  — True if drug found in FDA database
        standard_name  : str   — FDA standardised generic name
        brand_name     : str   — Brand name if available
        ndc_code       : str   — 10 or 11 digit National Drug Code
        dosage_form    : str   — e.g. "CAPSULE", "TABLET"
        route          : str   — e.g. "ORAL"
        strength       : str   — e.g. "40 MG/1"
        error          : str   — Present only if API call failed
    """
    print(f"\n  🔧 Tool: lookup_ndc('{drug_name}')")

    # Clean the drug name for search — remove OCR artifacts and
    # take only the first 2 meaningful words for better API matching
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', drug_name)
    clean_name = ' '.join(clean_name.split()[:2])   # "FLUoxetine HCI" → "FLUoxetine HCI"

    try:
        # Search by generic name first
        url      = (f"https://api.fda.gov/drug/ndc.json"
                    f"?search=generic_name:{clean_name}&limit=1")
        response = requests.get(url, timeout=API_TIMEOUT)

        # If generic search fails, try brand name
        if response.status_code == 404:
            url      = (f"https://api.fda.gov/drug/ndc.json"
                        f"?search=brand_name:{clean_name}&limit=1")
            response = requests.get(url, timeout=API_TIMEOUT)

        if response.status_code == 404:
            result = {
                "found": False, "standard_name": "", "brand_name": "",
                "ndc_code": "", "dosage_form": "", "route": "",
                "strength": "", "error": f"Drug '{drug_name}' not found in FDA NDC directory"
            }
            print(f"  ← Result: not found")
            return result

        response.raise_for_status()
        data    = response.json()
        results = data.get("results", [])

        if not results:
            result = {
                "found": False, "standard_name": "", "brand_name": "",
                "ndc_code": "", "dosage_form": "", "route": "",
                "strength": "", "error": "No results returned"
            }
            print(f"  ← Result: empty results")
            return result

        drug         = results[0]
        generic_name = drug.get("generic_name",  "")
        brand_name   = drug.get("brand_name",    "")
        ndc_code     = drug.get("product_ndc",   "")
        dosage_form  = drug.get("dosage_form",   "")
        route        = drug.get("route",         [])
        route_str    = route[0] if route else ""

        # Strength from active ingredients
        ingredients = drug.get("active_ingredients", [])
        strength    = ingredients[0].get("strength", "") if ingredients else ""

        result = {
            "found":         True,
            "standard_name": generic_name,
            "brand_name":    brand_name,
            "ndc_code":      ndc_code,
            "dosage_form":   dosage_form,
            "route":         route_str,
            "strength":      strength,
            "error":         ""
        }
        print(f"  ← Result: {generic_name} | NDC: {ndc_code} | {dosage_form} | {strength}")
        return result

    except requests.Timeout:
        result = {
            "found": False, "standard_name": "", "brand_name": "",
            "ndc_code": "", "dosage_form": "", "route": "",
            "strength": "", "error": "openFDA API timeout"
        }
        print(f"  ← Result: API timeout")
        return result

    except Exception as e:
        result = {
            "found": False, "standard_name": "", "brand_name": "",
            "ndc_code": "", "dosage_form": "", "route": "",
            "strength": "", "error": str(e)
        }
        print(f"  ← Result: error — {e}")
        return result


# ── Tool 3: validate_field_value ──────────────────────────────────────────────

@tool
def validate_field_value(field_name: str, value: str) -> dict:
    """
    Validate that an extracted field value matches US healthcare format rules.

    Call this tool when a value looks suspicious — for example:
    - A single word like 'Other' or 'Prescribed' where a name is expected
    - An NPI that doesn't look like 10 digits
    - A DEA number that doesn't match the standard pattern
    - A date with a 5-digit year (OCR error)
    - A strength value without a unit like MG or MCG

    Also call this to validate values the AI finds before adding them
    to found_fields — confirm the value is plausible before accepting it.

    Args:
        field_name: The field being validated. Examples:
                    "prescriber_npi", "dea_number", "dob",
                    "date_written", "strength", "refills"
        value:      The extracted value string to validate.

    Returns dict with:
        valid   : bool — True if value passes all rules for this field
        reason  : str  — What rule passed or failed
        suggest : str  — What a valid value should look like (if invalid)
    """
    print(f"\n  🔧 Tool: validate_field_value('{field_name}', '{value}')")

    v = str(value).strip()

    # ── NPI — must be exactly 10 digits ───────────────────────────────
    if field_name in ("prescriber_npi", "pharmacy_npi", "npi"):
        if re.fullmatch(r'\d{10}', v):
            result = {"valid": True,  "reason": "NPI is exactly 10 digits", "suggest": ""}
        else:
            result = {"valid": False, "reason": f"NPI must be 10 digits, got '{v}'",
                      "suggest": "Example: 1699091280"}

    # ── DEA — 2 uppercase letters + 7 digits ──────────────────────────
    elif field_name == "dea_number":
        if re.fullmatch(r'[A-Z]{2}\d{7}', v):
            result = {"valid": True,  "reason": "DEA format valid (2 letters + 7 digits)", "suggest": ""}
        else:
            result = {"valid": False, "reason": f"DEA must be 2 letters + 7 digits, got '{v}'",
                      "suggest": "Example: FW6804935"}

    # ── DOB — must be a past date, not future ─────────────────────────
    elif field_name == "dob":
        from datetime import date
        try:
            from dateutil import parser as dp
            parsed = dp.parse(v)
            if parsed.date() < date.today():
                result = {"valid": True,  "reason": "DOB is a valid past date", "suggest": ""}
            else:
                result = {"valid": False, "reason": f"DOB '{v}' is today or in the future",
                          "suggest": "DOB must be a past date"}
        except Exception:
            result = {"valid": False, "reason": f"Cannot parse '{v}' as a date",
                      "suggest": "Expected format: YYYY-MM-DD or MM/DD/YYYY"}

    # ── Date written — year must be exactly 4 digits ──────────────────
    elif field_name == "date_written":
        year_match = re.search(r'\b(\d+)\b', v.split('/')[-1] if '/' in v else v)
        if year_match and len(year_match.group(1)) == 4:
            result = {"valid": True,  "reason": "date_written has valid 4-digit year", "suggest": ""}
        elif year_match and len(year_match.group(1)) > 4:
            result = {"valid": False, "reason": f"Year has {len(year_match.group(1))} digits — OCR error",
                      "suggest": "Ask human to confirm the correct date"}
        else:
            result = {"valid": False, "reason": f"Cannot determine year in '{v}'",
                      "suggest": "Expected format: MM/DD/YYYY"}

    # ── Strength — number + valid unit ────────────────────────────────
    elif field_name == "strength":
        if re.search(r'\d+(?:\.\d+)?\s*(?:MG|MCG|ML|G|UNITS?)\b', v, re.IGNORECASE):
            result = {"valid": True,  "reason": "Strength has numeric value and valid unit", "suggest": ""}
        else:
            result = {"valid": False, "reason": f"'{v}' missing numeric value or valid unit (MG/MCG/ML/G)",
                      "suggest": "Example: 40 MG"}

    # ── Refills — 0 to 12 ─────────────────────────────────────────────
    elif field_name == "refills":
        if re.fullmatch(r'\d+', v) and 0 <= int(v) <= 12:
            result = {"valid": True,  "reason": f"Refills value {v} is in valid range 0–12", "suggest": ""}
        else:
            result = {"valid": False, "reason": f"Refills must be a number 0–12, got '{v}'",
                      "suggest": "Example: 0, 1, 2, or 3"}

    # ── Prescriber/patient name — must have 2+ parts, not a label ─────
    elif field_name in ("prescriber_name", "patient_name"):
        label_words = {"other", "prescribed", "product", "none", "na",
                       "unknown", "dispensed", "patient", "prescriber"}
        words = v.lower().split()
        if len(words) >= 2 and not any(w in label_words for w in words):
            result = {"valid": True,  "reason": "Name has 2+ parts and is not a form label", "suggest": ""}
        elif len(words) < 2:
            result = {"valid": False, "reason": f"'{v}' looks like a single word — may be a form label",
                      "suggest": "A real name should have at least first and last name"}
        else:
            result = {"valid": False, "reason": f"'{v}' matches known form label vocabulary",
                      "suggest": "Look for the actual patient/prescriber name in the document"}

    # ── Quantity — positive integer ────────────────────────────────────
    elif field_name == "qty_prescribed":
        if re.fullmatch(r'\d+', v) and int(v) > 0:
            result = {"valid": True,  "reason": f"Quantity {v} is a valid positive integer", "suggest": ""}
        else:
            result = {"valid": False, "reason": f"Quantity must be a positive integer, got '{v}'",
                      "suggest": "Example: 30"}

    # ── Unknown field — basic non-empty check ──────────────────────────
    else:
        if v and v.lower() not in {"none", "na", "n/a", "unknown", ""}:
            result = {"valid": True,  "reason": f"Value '{v}' is non-empty", "suggest": ""}
        else:
            result = {"valid": False, "reason": f"Value '{v}' is empty or placeholder",
                      "suggest": "Look for actual value in the document"}

    print(f"  ← Result: valid={result['valid']} | {result['reason']}")
    return result