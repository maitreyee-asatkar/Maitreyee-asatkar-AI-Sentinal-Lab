"""Framework-neutral interpretability primitives for Phase 3."""

from dataclasses import dataclass, field

@dataclass
class ActivationRecord:
    layer: int
    shape: tuple[int, ...]
    mean: float
    std: float

@dataclass
class InterpretationResult:
    model_id: str
    records: list[ActivationRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

def summarize_tensor(layer: int, tensor) -> ActivationRecord:
    detached = tensor.detach().float().cpu()
    return ActivationRecord(
        layer=layer,
        shape=tuple(detached.shape),
        mean=float(detached.mean().item()),
        std=float(detached.std().item()),
    )
