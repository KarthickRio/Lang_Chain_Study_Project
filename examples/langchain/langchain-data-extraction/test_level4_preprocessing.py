"""
Level 4 test — full preprocessing path (no LangGraph)

Manually chains:
    pdf_ingestion_node → native_preprocessing_node
                       → scanned_preprocessing_node

Run with:
    python test_level4_preprocessing.py

What this checks:
    1. State grows correctly through each node
    2. Doc type is detected correctly for your native PDF
    3. Fields are extracted and missing_fields is populated
    4. Confidence score is reasonable
    5. Scanned path runs without crashing (OCR output checked)
"""

from pdf_ingestion_node import pdf_ingestion_node, route_after_ingestion
from native_preprocessing_node import native_preprocessing_node
from scanned_preprocessing_node import scanned_preprocessing_node, route_after_preprocessing

NATIVE_PDF_PATH  = r"D:\langdb-samples\examples\langchain\langchain-data-extraction\Fake_Medical_Report.pdf"  
SCANNED_PDF_PATH = r"D:\\langdb-samples\\examples\\langchain\\langchain-data-extraction\\lydia_davila_11091940_20260211151053_refill_authorization_request.pdf .pdf"


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_initial_state(pdf_path: str) -> dict:
    return {
        "messages":       [],
        "pdf_path":       pdf_path,
        "error_messages": [],
    }


def print_state_snapshot(label: str, state: dict):
    """Print only the fields that matter at this stage."""
    print(f"\n{'─'*55}")
    print(f"  STATE SNAPSHOT: {label}")
    print(f"{'─'*55}")

    fields_to_show = [
        "pdf_type", "page_count", "doc_type",
        "doc_type_confidence", "confidence_score",
        "extracted_fields", "missing_fields",
        "current_phase", "error_messages",
    ]
    for f in fields_to_show:
        val = state.get(f, "— not set —")
        # Truncate long values for readability
        if isinstance(val, str) and len(val) > 80:
            val = val[:80] + "..."
        if isinstance(val, dict) and len(str(val)) > 200:
            val = {k: v for k, v in list(val.items())[:5]}
            val["..."] = "(truncated)"
        print(f"  {f:<25}: {val}")


def _print_checks(checks: dict):
    print()
    all_passed = True
    for label, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {label}")
        if not passed:
            all_passed = False
    print()
    if all_passed:
        print("  ✅ All checks passed\n")
    else:
        print("  ❌ Some checks failed — review output above\n")
    return all_passed


# ── Test 1: Native full path ──────────────────────────────────────────────────

def test_native_full_path():
    print("\n" + "="*55)
    print("  TEST: Native PDF — full preprocessing path")
    print("="*55)

    # Step 1 — ingestion
    state = make_initial_state(NATIVE_PDF_PATH)
    state = {**state, **pdf_ingestion_node(state)}
    print_state_snapshot("After ingestion", state)

    # Step 2 — check routing decision
    route = route_after_ingestion(state)
    print(f"\n  route_after_ingestion → '{route}'")

    # Step 3 — preprocessing
    assert route == "native_preprocessing", f"Expected native_preprocessing, got {route}"
    state = {**state, **native_preprocessing_node(state)}
    print_state_snapshot("After native preprocessing", state)

    # Step 4 — check routing after preprocessing
    route2 = route_after_preprocessing(state)
    print(f"\n  route_after_preprocessing → '{route2}'")

    # ── Assertions ────────────────────────────────────────────────────
    checks = {
        "doc_type is set":             state.get("doc_type") not in [None, "unknown"],
        "extracted_fields not empty":  len(state.get("extracted_fields", {})) > 0,
        "confidence_score > 0":        state.get("confidence_score", 0) > 0,
        "missing_fields is a list":    isinstance(state.get("missing_fields"), list),
        "no errors":                   state.get("error_messages") == [],
        "next route is valid":         route2 in ["ai_extraction", "synthesis"],
    }
    _print_checks(checks)

    # ── What the AI will receive ───────────────────────────────────────
    print("  What the AI extraction node will receive:")
    print(f"    doc_type        : {state.get('doc_type')}")
    print(f"    extracted_fields: {state.get('extracted_fields')}")
    print(f"    missing_fields  : {state.get('missing_fields')}")
    print(f"    confidence_score: {state.get('confidence_score')}")
    print(f"    → next node     : {route2}")


# ── Test 2: Scanned full path ─────────────────────────────────────────────────

def test_scanned_full_path():
    print("\n" + "="*55)
    print("  TEST: Scanned PDF — full preprocessing path")
    print("="*55)

    # Step 1 — ingestion
    state = make_initial_state(SCANNED_PDF_PATH)
    state = {**state, **pdf_ingestion_node(state)}
    print_state_snapshot("After ingestion", state)

    route = route_after_ingestion(state)
    print(f"\n  route_after_ingestion → '{route}'")

    # Step 2 — scanned preprocessing (includes OCR)
    assert route == "scanned_preprocessing", f"Expected scanned_preprocessing, got {route}"

    print("\n  Running OCR (this takes 10–30 seconds for 8 pages)...")
    state = {**state, **scanned_preprocessing_node(state)}
    print_state_snapshot("After scanned preprocessing", state)

    route2 = route_after_preprocessing(state)
    print(f"\n  route_after_preprocessing → '{route2}'")

    # ── Assertions ────────────────────────────────────────────────────
    checks = {
        "raw_text populated by OCR":   len(state.get("raw_text", "")) > 50,
        "doc_type is set":             state.get("doc_type") is not None,
        "confidence_score >= 0":       state.get("confidence_score", -1) >= 0,
        "missing_fields is a list":    isinstance(state.get("missing_fields"), list),
        "no errors":                   state.get("error_messages") == [],
        "next route is valid":         route2 in ["ai_extraction", "synthesis"],
    }
    _print_checks(checks)

    print("  OCR raw text sample (first 300 chars):")
    print(f"    {repr(state.get('raw_text', '')[:300])}")
    print(f"\n  What the AI extraction node will receive:")
    print(f"    doc_type        : {state.get('doc_type')}")
    print(f"    extracted_fields: {state.get('extracted_fields')}")
    print(f"    missing_fields  : {state.get('missing_fields')}")
    print(f"    confidence_score: {state.get('confidence_score')}")
    print(f"    → next node     : {route2}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_native_full_path()
    test_scanned_full_path()