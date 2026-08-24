from __future__ import annotations
import csv, json, random
from pathlib import Path
import torch
from torch.optim import AdamW
from q_tomo.train import load_model_from_checkpoint, evaluate_groups
from q_tomo.data import RuleMemoryCorpus, build_corpus
from q_tomo.runtime import resolve_device, seed_everything, autocast_context
from q_tomo.evaluate import batch_statistics

@torch.inference_mode()
def _subset_metrics(model, examples, corpus, device, batch_size):
    total_loss = 0.0; correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        stats = batch_statistics(model, batch, corpus, device)
        total_loss += stats["nll"].sum().item(); correct += stats["correct"].sum().item()
    return {"count": len(examples), "loss": total_loss / max(1, len(examples)), "accuracy": correct / max(1, len(examples))}

def run_interventions(checkpoint: str|Path, features: str|Path, output_dir: str|Path, fraction: float=.2, steps: int=1000, seed: int=42, learning_rate: float|None=None):
    seed_everything(seed); ck=Path(checkpoint).resolve(); out=Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    device=resolve_device("auto"); model, cfg=load_model_from_checkpoint(ck, device); corpus=build_corpus(cfg.data)
    rows=list(csv.DictReader(Path(features).open(encoding="utf-8"))); rows=[r for r in rows if r["group"] in {"generalized","memorized"} and r["base_correct"].lower()=="true"]
    mem=[r for r in rows if r["group"]=="memorized"]; k=max(1,round(len(mem)*fraction)); rng=random.Random(seed)
    strategies={"tomography":sorted(mem,key=lambda r:float(r["q_fragility"]),reverse=True)[:k],"confidence":sorted(mem,key=lambda r:float(r["base_margin"]))[:k]}
    sh=mem[:]; rng.shuffle(sh); strategies["random"]=sh[:k]
    baseline=evaluate_groups(model,corpus,device,cfg.probe.batch_size); lr = learning_rate if learning_rate is not None else cfg.train.learning_rate
    results={"baseline":baseline,"fraction":fraction,"steps":steps,"learning_rate":lr}
    for name,chosen in strategies.items():
        excluded={(int(r["relation"]),int(r["key"])) for r in chosen}
        members=[e for e in corpus.eval_examples if e.is_member and (e.relation,e.key) not in excluded]
        repeated=(members*cfg.data.exposures_per_pair); rng.shuffle(repeated)
        deleted = [e for e in corpus.examples_for_group("memorized") if (e.relation,e.key) in excluded]
        retained = [e for e in corpus.examples_for_group("memorized") if (e.relation,e.key) not in excluded]
        baseline_deleted = _subset_metrics(model, deleted, corpus, device, cfg.probe.batch_size)
        baseline_retained = _subset_metrics(model, retained, corpus, device, cfg.probe.batch_size)
        opt=AdamW(model.parameters(),lr=lr,weight_decay=cfg.train.weight_decay); model.train()
        for step in range(steps):
            batch=[repeated[(step*cfg.probe.batch_size+i)%len(repeated)] for i in range(cfg.probe.batch_size)]
            opt.zero_grad(set_to_none=True)
            with autocast_context(device,cfg.train.precision):
                input_ids=torch.tensor([e.input_ids for e in batch],dtype=torch.long,device=device)
                targets=torch.tensor([corpus.vocab.encode_number(e.value) for e in batch],dtype=torch.long,device=device)
                logits=model(input_ids)["logits"][:, list(corpus.prediction_positions), :].float()
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
            loss.backward(); opt.step()
        metrics=evaluate_groups(model,corpus,device,cfg.probe.batch_size)
        deleted_after = _subset_metrics(model, deleted, corpus, device, cfg.probe.batch_size)
        retained_after = _subset_metrics(model, retained, corpus, device, cfg.probe.batch_size)
        forgetting = baseline_deleted["accuracy"] - deleted_after["accuracy"]
        collateral = baseline["generalized"]["accuracy"] - metrics["generalized"]["accuracy"]
        results[name]={"excluded":k,"metrics":metrics,"deleted_memories":{"before":baseline_deleted,"after":deleted_after},"retained_memories":{"before":baseline_retained,"after":retained_after},"forgetting_gain":forgetting,"generalization_collateral":collateral,"forgetting_efficiency":forgetting / max(0.01, max(0.0, collateral))}
        model, _=load_model_from_checkpoint(ck,device)
    path=out/"analysis.json"; path.write_text(json.dumps(results,indent=2),encoding="utf-8"); return path
