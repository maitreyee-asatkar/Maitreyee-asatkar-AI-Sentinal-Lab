import re
from typing import Any
from evaluator.corpus import DEFAULT_CASES

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def evaluate_text(text: str) -> dict[str, Any]:
    normalized = _normalize(text)
    findings = []

    for case in DEFAULT_CASES:
        matched = [signal for signal in case["signals"] if signal.lower() in normalized]
        if matched:
            findings.append({
                "case_id": case["id"],
                "category": case["category"],
                "owasp": case["owasp"],
                "severity": case["severity"],
                "matched_signals": matched,
            })

    score = min(100, sum(
        {"critical": 40, "high": 25, "medium": 12, "low": 5}.get(f["severity"], 0)
        for f in findings
    ))

    return {
        "risk_score": score,
        "finding_count": len(findings),
        "findings": findings,
        "recommendation": (
            "Review the matched security controls and add a regression test."
            if findings else
            "No known MVP attack signals matched this input."
        ),
    }
