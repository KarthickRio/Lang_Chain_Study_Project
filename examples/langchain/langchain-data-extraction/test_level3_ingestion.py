"""
Level 3 test — pdf_ingestion_node

Run with:
    python test_level3_ingestion.py

What this checks:
    1. pdf_type is detected correctly (native vs scanned)
    2. raw_text is not empty for native PDFs
    3. page_count is correct
    4. First 500 chars of raw_text look sensible (no garbage)
    5. Error handling works for a bad file path
"""

from pdf_ingestion_node import pdf_ingestion_node

# ── Helpers ──────────────────────────────────────────────────────────────

def make_initial_state(pdf_path: str) -> dict:
    """
    Minimal state dict — only what ingestion_node needs.
    Simulates what LangGraph would pass in.
    """
    return {
        "messages":      [],
        "pdf_path":      pdf_path,
        "error_messages": [],
    }


def print_result(label: str, result: dict):
    print(f"\n{'='*55}")
    print(f"  TEST: {label}")
    print(f"{'='*55}")
    print(f"  pdf_type    : {result.get('pdf_type')}")
    print(f"  page_count  : {result.get('page_count')}")
    print(f"  raw_text len: {len(result.get('raw_text', ''))} chars")
    print(f"  errors      : {result.get('error_messages')}")

    raw = result.get("raw_text", "")
    if raw.strip():
        print(f"\n  --- First 500 chars of raw_text ---")
        print(f"  {repr(raw[:500])}")
    else:
        print(f"\n  raw_text is EMPTY")

    msgs = result.get("messages", [])
    if msgs:
        print(f"\n  Node message: {msgs[-1].content}")


# ── Test cases ────────────────────────────────────────────────────────────

def test_native_pdf():
    """Replace the path with your actual native PDF path"""
    path   = r"D:\langdb-samples\examples\langchain\langchain-data-extraction\Fake_Medical_Report.pdf"          # <-- change this
    state  = make_initial_state(path)
    result = pdf_ingestion_node(state)
    print_result("Native PDF", result)

    # Assertions — will print PASS or FAIL clearly
    checks = {
        "pdf_type is native"   : result.get("pdf_type") == "native",
        "raw_text not empty"   : len(result.get("raw_text", "")) > 100,
        "page_count > 0"       : result.get("page_count", 0) > 0,
        "no errors"            : result.get("error_messages") == [],
    }
    _print_checks(checks)


def test_scanned_pdf():
    """Replace the path with your actual scanned PDF path"""
    path   = r"D:\langdb-samples\examples\langchain\langchain-data-extraction\lydia_davila_11091940_20260211151053_refill_authorization_request.pdf .pdf"         # <-- change this
    state  = make_initial_state(path)
    result = pdf_ingestion_node(state)
    print_result("Scanned PDF", result)

    checks = {
        "pdf_type is scanned"  : result.get("pdf_type") == "scanned",
        "page_count > 0"       : result.get("page_count", 0) > 0,
        "no errors"            : result.get("error_messages") == [],
    }
    _print_checks(checks)


def test_bad_path():
    """Verify error handling when file doesn't exist"""
    state  = make_initial_state("nonexistent/fake.pdf")
    result = pdf_ingestion_node(state)
    print_result("Bad path (error handling)", result)

    checks = {
        "pdf_type is unknown"  : result.get("pdf_type") == "unknown",
        "error captured"       : len(result.get("error_messages", [])) > 0,
        "page_count is 0"      : result.get("page_count") == 0,
    }
    _print_checks(checks)


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
        print("  ✅ All checks passed")
    else:
        print("  ❌ Some checks failed — review output above")


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_native_pdf()
    test_scanned_pdf()
    test_bad_path()