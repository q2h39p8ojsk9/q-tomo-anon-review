from __future__ import annotations

import argparse
import json
from pathlib import Path

from q_tomo.config import ExperimentConfig
from q_tomo.controls import matched_noise_control
from q_tomo.data import RuleMemoryCorpus, build_corpus
from q_tomo.model import TomographyTransformer
from q_tomo.probe import probe_checkpoint
from q_tomo.robustness import robust_analysis
from q_tomo.train import train_experiment
from q_tomo.palm import palm_plan, causal_sweep, ranking_report, temporal_probe
from q_tomo.unlearning import run_unlearning


def _load_with_overrides(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.load(args.config)
    if args.seed is not None:
        config.train.seed = args.seed
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    return config


def _describe(config_path: str) -> None:
    config = ExperimentConfig.load(config_path)
    corpus = build_corpus(config.data)
    config.model.vocab_size = corpus.vocab.size
    config.model.max_seq_len = max(config.model.max_seq_len, len(corpus.eval_examples[0].input_ids))
    model = TomographyTransformer(config.model)
    print(json.dumps({"architecture": model.architecture_dict(), "corpus": corpus.metadata()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="q-tomo", description="Quantization tomography research toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="show resolved model and corpus statistics")
    describe.add_argument("--config", required=True)

    train = subparsers.add_parser("train", help="train a rule-memory model")
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--output-dir")

    probe = subparsers.add_parser("probe", help="run quantization tomography on a checkpoint")
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--output-dir")

    run = subparsers.add_parser("run", help="train and immediately probe")
    run.add_argument("--config", required=True)
    run.add_argument("--seed", type=int)
    run.add_argument("--output-dir")

    robust = subparsers.add_parser("robust", help="run reviewer-grade analysis on an existing feature CSV")
    robust.add_argument("--features", required=True)
    robust.add_argument("--output")
    robust.add_argument("--seed", type=int, default=0)

    noise = subparsers.add_parser("noise-control", help="run MSE-matched Gaussian perturbation control")
    noise.add_argument("--checkpoint", required=True)
    noise.add_argument("--output-dir")

    plan = subparsers.add_parser("palm-plan", help="write a PALM experiment plan without running jobs")
    plan.add_argument("--checkpoint", required=True); plan.add_argument("--features", required=True); plan.add_argument("--output-dir", required=True)
    sweep = subparsers.add_parser("palm-sweep", help="run PALM causal deletion budget/seed sweep")
    sweep.add_argument("--checkpoint", required=True); sweep.add_argument("--features", required=True); sweep.add_argument("--output-dir", required=True)
    sweep.add_argument("--budgets", nargs="+", type=float, default=[.05,.10,.20,.30,.50]); sweep.add_argument("--seeds", nargs="+", type=int, default=[42,43,44]); sweep.add_argument("--steps", type=int, default=5000)
    sweep.add_argument("--learning-rate", type=float)
    rank = subparsers.add_parser("palm-rank", help="compare PALM selection rankers")
    rank.add_argument("--features", required=True); rank.add_argument("--output", required=True); rank.add_argument("--budgets", nargs="+", type=float)
    temporal = subparsers.add_parser("palm-temporal", help="probe a checkpoint glob over training")
    temporal.add_argument("--checkpoint-glob", required=True); temporal.add_argument("--output-dir", required=True)
    unlearn = subparsers.add_parser("palm-unlearn", help="run retain-plus-random-target memory unlearning")
    unlearn.add_argument("--checkpoint", required=True); unlearn.add_argument("--features", required=True); unlearn.add_argument("--output-dir", required=True)
    unlearn.add_argument("--fraction", type=float, default=.2); unlearn.add_argument("--steps", type=int, default=1000); unlearn.add_argument("--seed", type=int, default=42)
    unlearn.add_argument("--learning-rate", type=float, default=2.5e-5); unlearn.add_argument("--forget-weight", type=float, default=1.0)

    pretrained_train = subparsers.add_parser("pretrained-train", help="inject controlled memories into a pretrained LM with LoRA")
    pretrained_train.add_argument("--config", required=True)
    pretrained_train.add_argument("--seed", type=int)
    pretrained_train.add_argument("--output-dir")
    pretrained_probe = subparsers.add_parser("pretrained-probe", help="probe a merged pretrained LoRA run")
    pretrained_probe.add_argument("--run-dir", required=True)
    pretrained_probe.add_argument("--output-dir")
    pretrained_decomposition = subparsers.add_parser("pretrained-decomposition", help="compare base-only, LoRA-only, and merged quantization")
    pretrained_decomposition.add_argument("--run-dir", required=True)
    pretrained_decomposition.add_argument("--output-dir")
    pretrained_decomposition_aggregate = subparsers.add_parser("pretrained-decomposition-aggregate", help="aggregate decomposition metrics across runs")
    pretrained_decomposition_aggregate.add_argument("--run-dirs", nargs="+", required=True)
    pretrained_decomposition_aggregate.add_argument("--output-dir", required=True)
    pretrained_effective_delta = subparsers.add_parser("pretrained-effective-delta", help="quantize effective LoRA delta weights")
    pretrained_effective_delta.add_argument("--run-dir", required=True)
    pretrained_effective_delta.add_argument("--output-dir")
    pretrained_scope_transfer = subparsers.add_parser("pretrained-scope-transfer", help="leave-one-seed-out transfer across perturbation scopes")
    pretrained_scope_transfer.add_argument("--run-dirs", nargs="+", required=True)
    pretrained_scope_transfer.add_argument("--output-dir", required=True)
    pretrained_matched_noise = subparsers.add_parser("pretrained-matched-noise", help="run MSE-matched Gaussian controls on pretrained scopes")
    pretrained_matched_noise.add_argument("--run-dir", required=True)
    pretrained_matched_noise.add_argument("--output-dir")
    pretrained_noise_transfer = subparsers.add_parser("pretrained-noise-transfer", help="compare quantization and matched noise across held-out seeds")
    pretrained_noise_transfer.add_argument("--run-dirs", nargs="+", required=True)
    pretrained_noise_transfer.add_argument("--output-dir", required=True)
    pretrained_prompt_probe = subparsers.add_parser("pretrained-prompt-probe", help="probe tomography under unseen prompt templates")
    pretrained_prompt_probe.add_argument("--run-dir", required=True)
    pretrained_prompt_probe.add_argument("--output-dir")
    pretrained_prompt_transfer = subparsers.add_parser("pretrained-prompt-transfer", help="test canonical-trained detectors on unseen prompts and seeds")
    pretrained_prompt_transfer.add_argument("--run-dirs", nargs="+", required=True)
    pretrained_prompt_transfer.add_argument("--output-dir", required=True)
    reviewer_controls = subparsers.add_parser("pretrained-reviewer-controls", help="four-class controls and cluster-bootstrap transfer intervals")
    reviewer_controls.add_argument("--run-dirs", nargs="+", required=True)
    reviewer_controls.add_argument("--output-dir", required=True)
    reviewer_controls.add_argument("--bootstrap-samples", type=int, default=2000)
    reviewer_controls.add_argument("--seed", type=int, default=1729)
    counterfactual = subparsers.add_parser("counterfactual-memorization", help="balanced inclusion/exclusion training ensemble")
    counterfactual.add_argument("--config", required=True)
    counterfactual.add_argument("--output-dir", required=True)
    counterfactual.add_argument("--reference-features")
    counterfactual.add_argument("--models", type=int, default=12)
    counterfactual.add_argument("--steps", type=int, default=4000)
    counterfactual.add_argument("--seed", type=int, default=31415)
    pretrained_unlearn = subparsers.add_parser("pretrained-unlearn", help="targeted PALM-style rewriting on a pretrained LoRA run")
    pretrained_unlearn.add_argument("--run-dir", required=True)
    pretrained_unlearn.add_argument("--features", required=True)
    pretrained_unlearn.add_argument("--output-dir")
    pretrained_unlearn.add_argument("--fraction", type=float, default=0.20)
    pretrained_unlearn.add_argument("--steps", type=int, default=100)
    pretrained_unlearn.add_argument("--seed", type=int, default=42)
    pretrained_unlearn.add_argument("--learning-rate", type=float, default=2e-4)
    pretrained_unlearn.add_argument("--kl-weight", type=float, default=1.0)
    pretrained_unlearn.add_argument("--strategies", nargs="+", choices=["tomography", "confidence", "random"])
    pretrained_unlearn.add_argument("--candidate-scope", choices=["memorized", "train_members", "eval_split"], default="eval_split")
    pretrained_unlearn_aggregate = subparsers.add_parser("pretrained-unlearn-aggregate", help="aggregate corrected pretrained unlearning sweeps")
    pretrained_unlearn_aggregate.add_argument("--run-dirs", nargs="+", required=True)
    pretrained_unlearn_aggregate.add_argument("--output-dir", required=True)
    pretrained_run = subparsers.add_parser("pretrained-run", help="train and probe a controlled pretrained LM")
    pretrained_run.add_argument("--config", required=True)
    pretrained_run.add_argument("--seed", type=int)
    pretrained_run.add_argument("--output-dir")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "describe":
        _describe(args.config)
    elif args.command == "train":
        checkpoint = train_experiment(_load_with_overrides(args))
        print(checkpoint)
    elif args.command == "probe":
        print(probe_checkpoint(args.checkpoint, args.output_dir))
    elif args.command == "run":
        checkpoint = train_experiment(_load_with_overrides(args))
        print(probe_checkpoint(checkpoint))
    elif args.command == "robust":
        print(robust_analysis(args.features, args.output, args.seed))
    elif args.command == "noise-control":
        print(matched_noise_control(args.checkpoint, args.output_dir))
    elif args.command == "palm-plan":
        print(palm_plan(args.checkpoint, args.features, args.output_dir))
    elif args.command == "palm-sweep":
        print(causal_sweep(args.checkpoint, args.features, args.output_dir, args.budgets, args.seeds, args.steps, args.learning_rate))
    elif args.command == "palm-rank":
        print(ranking_report(args.features, args.output, args.budgets))
    elif args.command == "palm-temporal":
        print(temporal_probe(args.checkpoint_glob, args.output_dir))
    elif args.command == "palm-unlearn":
        print(run_unlearning(args.checkpoint, args.features, args.output_dir, args.fraction, args.steps, args.seed, args.learning_rate, args.forget_weight))
    elif args.command in {"pretrained-train", "pretrained-run"}:
        from q_tomo.pretrained import PretrainedConfig, run_pretrained, train_pretrained
        config = PretrainedConfig.load(args.config)
        if args.seed is not None:
            config.seed = args.seed
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        result = run_pretrained(config) if args.command == "pretrained-run" else train_pretrained(config)
        print(result)
    elif args.command == "pretrained-probe":
        from q_tomo.pretrained import probe_pretrained
        print(probe_pretrained(args.run_dir, args.output_dir))
    elif args.command == "pretrained-decomposition":
        from q_tomo.pretrained import probe_pretrained_decomposition
        print(probe_pretrained_decomposition(args.run_dir, args.output_dir))
    elif args.command == "pretrained-decomposition-aggregate":
        from q_tomo.pretrained import aggregate_pretrained_decomposition
        print(aggregate_pretrained_decomposition(args.run_dirs, args.output_dir))
    elif args.command == "pretrained-effective-delta":
        from q_tomo.pretrained import probe_pretrained_effective_delta
        print(probe_pretrained_effective_delta(args.run_dir, args.output_dir))
    elif args.command == "pretrained-scope-transfer":
        from q_tomo.pretrained import cross_seed_pretrained_scope_transfer
        print(cross_seed_pretrained_scope_transfer(args.run_dirs, args.output_dir))
    elif args.command == "pretrained-matched-noise":
        from q_tomo.pretrained import probe_pretrained_matched_noise
        print(probe_pretrained_matched_noise(args.run_dir, args.output_dir))
    elif args.command == "pretrained-noise-transfer":
        from q_tomo.pretrained import cross_seed_pretrained_noise_control
        print(cross_seed_pretrained_noise_control(args.run_dirs, args.output_dir))
    elif args.command == "pretrained-prompt-probe":
        from q_tomo.pretrained import probe_pretrained_prompt_robustness
        print(probe_pretrained_prompt_robustness(args.run_dir, args.output_dir))
    elif args.command == "pretrained-prompt-transfer":
        from q_tomo.pretrained import cross_seed_pretrained_prompt_transfer
        print(cross_seed_pretrained_prompt_transfer(args.run_dirs, args.output_dir))
    elif args.command == "pretrained-reviewer-controls":
        from q_tomo.reviewer_controls import pretrained_reviewer_controls
        print(pretrained_reviewer_controls(
            args.run_dirs, args.output_dir, args.bootstrap_samples, args.seed
        ))
    elif args.command == "counterfactual-memorization":
        from q_tomo.counterfactual import run_counterfactual_ensemble
        print(run_counterfactual_ensemble(
            args.config, args.output_dir, args.reference_features,
            args.models, args.steps, args.seed,
        ))
    elif args.command == "pretrained-unlearn":
        from q_tomo.pretrained import targeted_unlearning_pretrained
        print(targeted_unlearning_pretrained(
            args.run_dir, args.features, args.output_dir, args.fraction, args.steps,
            args.seed, args.learning_rate, args.kl_weight, args.strategies,
            args.candidate_scope,
        ))
    elif args.command == "pretrained-unlearn-aggregate":
        from q_tomo.pretrained import aggregate_pretrained_unlearning
        print(aggregate_pretrained_unlearning(args.run_dirs, args.output_dir))


if __name__ == "__main__":
    main()
