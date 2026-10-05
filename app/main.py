from fastapi import FastAPI
from pydantic import BaseModel, Field
from evaluator.engine import evaluate_text
from evaluator.corpus import DEFAULT_CASES

app = FastAPI(
    title="AI Sentinel Lab",
    version="0.1.0",
    description="Local-first AI security evaluation API.",
)

class AssessmentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)

@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-sentinel-lab"}

@app.get("/cases")
def cases():
    return {"count": len(DEFAULT_CASES), "cases": DEFAULT_CASES}

@app.post("/assess")
def assess(request: AssessmentRequest):
    return evaluate_text(request.text)
