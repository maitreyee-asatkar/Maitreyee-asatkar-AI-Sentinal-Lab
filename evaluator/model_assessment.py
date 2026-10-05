from evaluator.engine import evaluate_text
from models.huggingface_adapter import HuggingFaceAdapter

def assess_model_prompt(prompt: str, adapter: HuggingFaceAdapter | None = None):
    adapter = adapter or HuggingFaceAdapter()
    generation = adapter.generate(prompt)

    return {
        "model_id": generation.model_id,
        "prompt": prompt,
        "response": generation.response,
        "prompt_assessment": evaluate_text(prompt),
        "response_assessment": evaluate_text(generation.response),
    }
