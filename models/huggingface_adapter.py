import os
from dataclasses import dataclass

@dataclass
class GenerationResult:
    prompt: str
    response: str
    model_id: str

class HuggingFaceAdapter:
    """Lazy-loading local Hugging Face model adapter."""

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("AI_SENTINEL_MODEL", "distilgpt2")
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.eval()

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 80) -> GenerationResult:
        import torch

        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = self._tokenizer.decode(output[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()

        return GenerationResult(prompt, response, self.model_id)
