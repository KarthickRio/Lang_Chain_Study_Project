"""
main.py

Entry point. Pass a PDF path as a command-line argument.

Usage:
    python main.py path/to/document.pdf

The agent runs the full pipeline and prints the final_output dict.
Set ANTHROPIC_API_KEY in your environment before running.

Windows PowerShell:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    python main.py "D:\\path\\to\\your.pdf"
"""

import sys
import json
from langchain_core.messages import HumanMessage
from state import MedicalFaxState, PDFType, ProcessingPhase, DocType
from agent import create_agent


def run(pdf_path: str):
    agent = create_agent()

    initial_state = {
        "messages":            [HumanMessage(content=f"Process PDF: {pdf_path}")],
        "pdf_path":            pdf_path,
        "pdf_type":            PDFType.UNKNOWN,
        "raw_text":            "",
        "page_count":          0,
        "text_per_page":       [],
        "doc_type":            DocType.UNKNOWN,
        "doc_type_confidence": 0.0,
        "extracted_fields":    {},
        "missing_fields":      [],
        "confidence_score":    0.0,
        "ai_feedback":         [],
        "ai_refined_fields":   {},
        "validation_passed":   False,
        "current_phase":       ProcessingPhase.INGESTION,
        "error_messages":      [],
        "final_output":        None,
    }

    print(f"\n{'='*55}")
    print(f"  MEDICAL FAX AGENT")
    print(f"  PDF: {pdf_path}")
    print(f"{'='*55}")

    final_state = agent.invoke(initial_state)

    # ── Print final output ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL OUTPUT")
    print(f"{'='*55}")
    output = final_state.get("final_output")
    if output:
        print(json.dumps(output, indent=2, default=str))
    else:
        print("  No final output — check error_messages:")
        print(f"  {final_state.get('error_messages')}")

    return final_state


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        print('Example: python main.py "D:\\docs\\fax.pdf"')
        sys.exit(1)

    run(sys.argv[1])
    