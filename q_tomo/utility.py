from __future__ import annotations
import csv, json, random
from pathlib import Path

def run_utility(features: str | Path, output_dir: str | Path | None = None, seed: int = 42) -> Path:
    src = Path(features).resolve(); out = Path(output_dir).resolve() if output_dir else src.parent.parent / "utility_intervention"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    rows = [r for r in rows if r["group"] in {"generalized", "memorized"} and r.get("base_correct", "False").lower() == "true"]
    for r in rows:
        r["label"] = 1 if r["group"] == "memorized" else 0
        r["q_fragility"] = float(r["q_fragility"]); r["base_confidence"] = float(r["base_margin"])
    rng = random.Random(seed); n_mem = sum(r["label"] for r in rows); n = len(rows)
    results = {}
    for frac in (0.05, 0.10, 0.20, 0.30):
        k = max(1, round(frac*n))
        strategies = {
            "tomography": sorted(rows, key=lambda r:r["q_fragility"], reverse=True)[:k],
            "confidence": sorted(rows, key=lambda r:r["base_confidence"])[:k],
        }
        shuffled = rows[:]; rng.shuffle(shuffled); strategies["random"] = shuffled[:k]
        results[str(frac)] = {name: {"selected":k, "memorized_recall": sum(r["label"] for r in chosen)/max(1,n_mem), "memorized_precision": sum(r["label"] for r in chosen)/k} for name, chosen in strategies.items()}
    payload = {"source":str(src), "n_correct":n, "memorized":n_mem, "selection":results, "interpretation":"proxy utility: higher memorized recall at fixed budget identifies better candidates for targeted forgetting/compression"}
    path = out / "analysis.json"; path.write_text(json.dumps(payload, indent=2), encoding="utf-8"); return path
