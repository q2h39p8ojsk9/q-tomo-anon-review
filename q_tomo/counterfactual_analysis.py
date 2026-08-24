"""Post-hoc uncertainty and figure-ready summaries for counterfactual runs."""
from __future__ import annotations

import csv, json, random, math
from pathlib import Path
from statistics import fmean
from q_tomo.analysis import roc_auc

def _rank(values):
    order=sorted(range(len(values)), key=values.__getitem__); out=[0.0]*len(values); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        r=(i+j-1)/2
        for k in range(i,j): out[order[k]]=r
        i=j
    return out

def _rho(a,b):
    a,b=_rank(a),_rank(b); ma,mb=fmean(a),fmean(b)
    d=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/d if d else float('nan')

def _ci(values):
    values=sorted(values); return [values[max(0,int(.025*(len(values)-1)))], values[min(len(values)-1,int(.975*(len(values)-1)))]]

def summarize_counterfactual(run_dir, output=None, bootstrap_samples=4000, seed=271828):
    run=Path(run_dir); rows=list(csv.DictReader((run/'example_records.csv').open(encoding='utf-8')))
    relations=sorted({int(r['relation']) for r in rows}); rng=random.Random(seed)
    boot=[]
    for _ in range(bootstrap_samples):
        chosen=[rng.choice(relations) for _ in relations]
        sample=[r for rel in chosen for r in rows if int(r['relation'])==rel]
        cf=[float(r['counterfactual_memorization']) for r in sample]
        q=[float(r['q_fragility']) for r in sample]
        y=[r['group']=='memorized' for r in sample]
        boot.append((_rho(cf,q), roc_auc(y,cf), roc_auc(y,q)))
    result=json.loads((run/'analysis.json').read_text(encoding='utf-8'))
    result['relation_bootstrap']={
      'samples':bootstrap_samples, 'relation_count':len(relations), 'seed':seed,
      'spearman_ci':_ci([x[0] for x in boot]),
      'counterfactual_auc_ci':_ci([x[1] for x in boot]),
      'fragility_auc_ci':_ci([x[2] for x in boot]),
    }
    result['group_summary']['gap_memorized_minus_seen_rule']=(
      result['group_summary']['memorized']['counterfactual_memorization_mean']-
      result['group_summary']['seen_rule']['counterfactual_memorization_mean'])
    target=Path(output) if output else run/'analysis.json'
    target.write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    return target

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('run_dir'); p.add_argument('--output'); p.add_argument('--samples',type=int,default=4000)
    a=p.parse_args(); print(summarize_counterfactual(a.run_dir,a.output,a.samples))
