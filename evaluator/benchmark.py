import json
from pathlib import Path
from evaluator.corpus import DEFAULT_CASES
from evaluator.model_assessment import assess_model_prompt

def run_benchmark(output: str = "artifacts/model_benchmark.json"):
    results = []
    for case in DEFAULT_CASES:
        result = assess_model_prompt(case["prompt"])
        results.append({
            "case_id": case["id"],
            "category": case["category"],
            "owasp": case["owasp"],
            "severity": case["severity"],
            "model_id": result["model_id"],
            "response": result["response"],
            "response_risk_score": result["response_assessment"]["risk_score"],
            "response_findings": result["response_assessment"]["findings"],
        })
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results

if __name__ == "__main__":
    data = run_benchmark()
    print(f"Completed {len(data)} security cases.")
