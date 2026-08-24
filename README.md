# Q-Tomo

Q-Tomo is an experimental framework for asking whether controlled weight
quantization can distinguish **memorized instances** from **generalized rules**.
It is designed for a single 8 GB RTX 3070 Ti and scales from approximately 5M
to 100M parameters.

The repository implements the first, falsifiable stage of the project:

1. train one causal transformer on a matched mixture of learnable rules and
   random associations;
2. keep only outputs that the full-precision model answers correctly;
3. repeatedly quantize the whole model or individual attention/MLP blocks with
   stochastic rounding;
4. build a per-example quantization survival signature; and
5. test whether that signature separates generalization from memorization
   beyond ordinary confidence features.

## Why these models and data

The primary model is a compact, dense decoder-only transformer using RoPE,
pre-RMSNorm, SwiGLU, bias-free projections, and optional per-head QK
normalization. These are conservative modern components shared by model
families such as Qwen3. At this scale, standard multi-head attention is kept by
default: grouped-query attention saves little at a nine-token context and would
unnecessarily change representational capacity. A GPT-2-style learned-position,
LayerNorm, GELU control is included to test architecture dependence.

The main corpus is deliberately **not** ordinary web text. Natural text does not
tell us whether an individual output was computed from a reusable rule or
recalled from training. `RuleMemoryCorpus` gives both labels by construction:

- primary rule relations copy a digit-composed key, a deliberately reliable
  compositional operation; an affine-permutation family is retained as a
  harder stress test;
- memory relations are uniformly random permutations;
- both use the same token grammar, sequence length, key split, exposure count,
  and exactly matched train/holdout output marginals;
- numbers are digit-tokenized, so rules can generalize compositionally rather
  than assigning an independent embedding to every integer;
- `generalized` examples are held-out rule inputs, while `memorized` examples
  are random associations observed during training.

Two stronger controls are also included. `mixed_copy_5m.json` places copied
rules and random exceptions inside the same relation tokens, eliminating
relation identity as a shortcut. `mixed_digitwise_5m.json` replaces COPY with
16 heterogeneous bijections; the current 5M model fails its generalization
readiness floor, so that configuration is a capability/curriculum target rather
than positive evidence.

For later external-validity experiments, use **TinyStories** for the 5M model
and a fixed, documented **FineWeb-Edu** sample for the 25M/100M models, injecting
random canaries before training. TinyStories was explicitly designed to make
sub-10M language models scientifically useful, while FineWeb-Edu is a strongly
filtered educational corpus. Neither replaces the controlled corpus; they test
whether its result transfers to natural language.

References:

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [TinyStories](https://arxiv.org/abs/2305.07759)
- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- [Pythia](https://github.com/EleutherAI/pythia)
- [Bits for Privacy](https://arxiv.org/abs/2512.15335)
- [CheckMIABench](https://arxiv.org/abs/2606.17464)

## Installation

Python 3.10+ and PyTorch 2.2+ are required. On Windows with a recent NVIDIA
driver:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Confirm the actual GPU runtime rather than assuming CUDA is active:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_available())"
```

## Quick start

Inspect exact parameter counts and corpus statistics:

```powershell
.\.venv\Scripts\python.exe -m q_tomo describe --config configs/pilot_5m.json
```

Run unit tests and the tiny end-to-end smoke experiment:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m q_tomo run --config configs/smoke.json
```

Run the decisive 5M pilot:

```powershell
.\.venv\Scripts\python.exe -m q_tomo run --config configs/pilot_5m.json
```

### Pretrained-model experiment

The workshop experiment injects controlled rule applications and matched
item-specific exceptions into Qwen3 using LoRA, a small trainable low-rank
weight update. The base-model parameters remain frozen during training. The
1.7B configuration was designed to fit on an 8 GB RTX 3070 Ti:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,pretrained]"
.\.venv\Scripts\python.exe -m q_tomo pretrained-run --config configs/pretrained_qwen3_17b.json --seed 42
```

Repeat the command with independent seeds and distinct output directories, for
example `--seed 43 --output-dir runs/pretrained/qwen3_17b_seed43`. The command
trains the LoRA adapter and then measures how each correctly answered example
responds to temporary weight perturbations. Checkpoints, downloaded model
weights, and generated results are intentionally excluded from version
control.

The CLI also exposes the decomposition, held-out prompt, matched-noise,
cross-seed transfer, reviewer-control, and targeted-unlearning analyses used in
the study. Run `python -m q_tomo --help` for the complete command list and
`python -m q_tomo <command> --help` for exact arguments.

Run reviewer-grade reanalysis and the MSE-matched perturbation control:

```powershell
.\.venv\Scripts\python.exe -m q_tomo robust --features runs/full/modern_seed42/tomography/features.csv --seed 42
.\.venv\Scripts\python.exe -m q_tomo noise-control --checkpoint runs/full/modern_seed42/checkpoint_best_ready.pt
```

The robust analysis uses leave-whole-relations-out CV, confidence matching,
relation-cluster bootstrap intervals, relation-label permutation tests,
bit-width ablations, and layer localization. The noise control substitutes
per-tensor Gaussian perturbations with RMS exactly matched to each quantization
condition.

The final report is written to
`runs/pilot_5m_copy/tomography/analysis.json`; per-example features are in
`features.csv`. Training checkpoints are retained so tomography can be repeated
with different bit widths without retraining.

During training, `checkpoint_best_ready.pt` tracks the checkpoint maximizing
the smaller of generalized and memorized validation accuracy. The `run`
command probes this checkpoint. This selection never looks at tomography AUROC,
so it prevents late lookup overfitting without selecting on the reported effect.

`configs/affine_stress_5m.json` is intentionally not the primary pilot. It asks
the model to infer relation-specific modular affine maps from examples; failure
to generalize on that task means those examples cannot be labeled as rule-based
for tomography. Treat it as a curriculum/model-capability stress test.

## Model presets

| Config | Architecture | Approximate parameters | Role |
|---|---:|---:|---|
| `pilot_5m.json` | modern, 288 × 5 | 5M | decisive pilot |
| `control_gpt2_5m.json` | GPT-2, 256 × 6 | 5M | architecture control |
| `scale_25m.json` | modern, 512 × 8 | 26M | scaling point |
| `scale_100m.json` | modern, 768 × 14 | 100M | upper local scale |

The 100M preset uses gradient accumulation to keep the optimizer and temporary
quantization copies comfortably inside 8 GB. Contexts in the controlled study
are tiny, so activation memory is not the limiting factor.

## What the kill test means

The analysis compares two cross-validated logistic probes among examples that
the unquantized model gets right:

- baseline: full-precision NLL and answer margin;
- combined: baseline plus quantization fragility, variability, survival, and
  layer-concentration features.

The configured pilot passes only when both groups have enough correct examples,
combined AUROC is at least 0.80, and tomography improves over confidence by at
least 0.05 AUROC. This is a research decision rule, not a privacy guarantee.

## First local result

On the RTX 3070 Ti, seed 42 reached the readiness criterion at the saved
step-1,000 checkpoint with 266 correct generalized examples and 1,153 correct
memorized examples. The preregistered-style kill test passed: confidence-only
CV AUROC was 0.702, combined confidence plus tomography was 0.852, and the
increment was +0.150. This is an encouraging single-seed pilot, not an ICML
result yet; the next requirement is replication across training seeds and the
GPT-2 control.

The affine stress-test run was also informative: it memorized every observed
pair but generalized at only 0.5%. Those examples therefore cannot honestly be
used as the "reasoned" class without a stronger arithmetic curriculum.

Subsequent controls narrow the interpretation. The within-relation benchmark
replicates positive incremental AUROC across three seeds (+0.141, +0.070,
+0.102), showing the signal is not created by relation-token identity. However,
MSE-matched Gaussian noise reproduces most of the effect. The defensible object
of study is therefore **parameter-perturbation tomography**, with quantization
as a convenient structured probe, rather than a quantization-exclusive effect.

## Experimental order

1. Run `smoke.json` only to validate software.
2. Run `pilot_5m.json` with three independent training seeds.
3. Run the GPT-2 control before scaling up.
4. If the incremental AUROC does not replicate, stop or reformulate the claim.
5. If it replicates, run 25M and 100M, then use Pythia/CheckMIABench for
   external validation.
