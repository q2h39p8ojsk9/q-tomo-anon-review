"""Reproducible PALM experiment plans and causal forgetting sweeps.

These commands only execute when explicitly invoked; generating a plan is cheap
and does not reserve the GPU.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from q_tomo.intervention import run_interventions


def palm_plan(checkpoint: str | Path, features: str | Path, output_dir: str | Path) -> Path:
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    plan = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "features": str(Path(features).resolve()),
        "experiments": [
            {"name": "causal_budget_seed_sweep", "budgets": [0.05, 0.10, 0.20, 0.30, 0.50], "seeds": [42, 43, 44], "steps": 5000},
            {"name": "ranking_baselines", "rankers": ["tomography", "confidence", "loss", "margin", "random"], "budgets": [0.10, 0.20, 0.30]},
            {"name": "temporal_probe", "checkpoint_glob": "checkpoints/step_*.pt", "probe": "stochastic+layerwise"},
            {"name": "localization", "scopes": ["embeddings", "early_attention", "early_mlp", "late_attention", "late_mlp"]},
            {"name": "external_memory", "format": "jsonl", "fields": ["text", "memory_id", "split", "is_member"], "splits": ["natural", "counterfactual", "heldout"]},
        ],
    }
    path = out / "palm_plan.json"; path.write_text(json.dumps(plan, indent=2), encoding="utf-8"); return path


def causal_sweep(checkpoint: str | Path, features: str | Path, output_dir: str | Path,
                 budgets: list[float], seeds: list[int], steps: int = 5000, learning_rate: float | None = None) -> Path:
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True); records = []
    for budget in budgets:
        for seed in seeds:
            target = out / f"budget_{budget:.2f}_seed_{seed}"
            artifact = run_interventions(checkpoint, features, target, budget, steps, seed, learning_rate)
            records.append({"budget": budget, "seed": seed, "steps": steps, "learning_rate": learning_rate, "artifact": str(artifact)})
    path = out / "sweep_manifest.json"; path.write_text(json.dumps(records, indent=2), encoding="utf-8"); return path


def ranking_report(features: str | Path, output: str | Path, budgets: list[float] | None = None, seed: int = 42) -> Path:
    src = Path(features).resolve(); out = Path(output).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    rows = [r for r in rows if r["group"] in {"generalized", "memorized"} and r["base_correct"].lower() == "true"]
    for r in rows:
        r["label"] = int(r["group"] == "memorized")
        r["loss"] = float(r["base_nll"]); r["margin"] = float(r["base_margin"]); r["tomography"] = float(r["q_fragility"])
    budgets = budgets or [0.10, 0.20, 0.30]; rng = random.Random(seed); report = []
    for frac in budgets:
        k = max(1, round(len(rows) * frac))
        candidates = {"tomography": sorted(rows, key=lambda r: r["tomography"], reverse=True), "loss": sorted(rows, key=lambda r: r["loss"], reverse=True), "margin": sorted(rows, key=lambda r: r["margin"]), "confidence": sorted(rows, key=lambda r: r["margin"]), "random": rows[:]}
        rng.shuffle(candidates["random"])
        for name, ordered in candidates.items():
            chosen = ordered[:k]; mem = sum(r["label"] for r in chosen)
            report.append({"budget": frac, "ranker": name, "selected": k, "memorized_precision": mem / k, "memorized_recall": mem / max(1, sum(r["label"] for r in rows))})
    out.write_text(json.dumps({"source": str(src), "results": report}, indent=2), encoding="utf-8"); return out


def temporal_probe(checkpoint_glob: str | Path, output_dir: str | Path) -> Path:
    from q_tomo.probe import probe_checkpoint
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True); records = []
    for checkpoint in sorted(Path(checkpoint_glob).parent.glob(Path(checkpoint_glob).name)):
        artifact = probe_checkpoint(checkpoint, out / checkpoint.stem)
        records.append({"checkpoint": str(checkpoint), "artifact": str(artifact)})
    path = out / "temporal_manifest.json"; path.write_text(json.dumps(records, indent=2), encoding="utf-8"); return path
