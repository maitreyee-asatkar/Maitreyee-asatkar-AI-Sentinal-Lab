"""Generate a dependency-free Markdown dashboard from research JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ARTIFACTS = Path("artifacts")


def load(name):
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    lines = [
        "# AI Sentinel Lab — Research Dashboard",
        "",
        "Generated from reproducible JSON artifacts. Missing experiments are shown as pending.",
        "",
    ]

    ablation = load("head_ablation.json")
    if ablation:
        lines += ["## Causal head-ablation replication", "", "| Head | Mean restoration | Positive pairs |", "|---|---:|---:|"]
        for row in ablation["aggregate_ranking"]:
            lines.append(
                f'| L{row["layer"]}:H{row["head"]} | {pct(row["mean_restoration_fraction"])} | '
                f'{row["positive_restoration_pairs"]}/{row["pair_count"]} |'
            )
        lines.append("")
    else:
        lines += ["## Causal head-ablation replication", "", "Pending.", ""]

    steering = load("activation_steering.json")
    if steering:
        lines += ["## Activation steering: security vs utility", "", "| Coefficient | Security restoration | Positive pairs | Clean utility drift (KL) | Tradeoff score |", "|---:|---:|---:|---:|---:|"]
        for row in steering["aggregate_ranking"]:
            lines.append(
                f'| {row["coefficient"]:.2f} | {pct(row["mean_security_restoration_fraction"])} | '
                f'{row["positive_restoration_pairs"]}/{row["pair_count"]} | '
                f'{row["mean_clean_utility_drift_kl"]:.6f} | {row["tradeoff_score"]:.6f} |'
            )
        lines.append("")
    else:
        lines += ["## Activation steering: security vs utility", "", "Pending.", ""]

    lines += [
        "## Interpretation guardrails",
        "",
        "- Attention and activation differences are descriptive candidate signals.",
        "- Head ablation and activation steering are causal only for the tested intervention.",
        "- Multi-pair replication improves evidence, but does not establish universal behavior.",
        "- Security gains should be considered together with clean-utility drift.",
        "",
    ]

    output = ARTIFACTS / "research_dashboard.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
