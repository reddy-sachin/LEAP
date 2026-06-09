import argparse

import torch


def load_config():
    """Parse command-line configuration for training/evaluation.

    Returns:
        argparse.Namespace containing runtime, data, optimization and plotting options.
    """
    parser = argparse.ArgumentParser(description="LEAP training/evaluation config")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-eval", dest="train_eval", action="store_true", help="Train and evaluate")
    mode.add_argument("--eval-only", dest="eval_only", action="store_true", help="Evaluate only from HF test split + pretrained artifacts")
    mode.add_argument("--plot-only", "--plot_only", dest="plot_only", action="store_true", help="Generate plot from existing results CSV")

    parser.add_argument("--hf-dataset-repo", type=str, default="reddysachin/LEAP_dataset")
    parser.add_argument("--hf-model-repo", type=str, default="reddysachin/LEAP")

    parser.add_argument("--output-dir", type=str, default="data/out")
    parser.add_argument("--weights-name", type=str, default="best_weights.pt")
    parser.add_argument("--final-weights-name", type=str, default="weights.pt")
    parser.add_argument("--history-name", type=str, default="training_history.csv")
    parser.add_argument("--results-name", type=str, default="test_results.csv")

    parser.add_argument("--n-epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--noise-level", type=float, default=1e-3)
    parser.add_argument("--w-par", type=float, default=0.5)
    parser.add_argument("--w-perp", type=float, default=1.0)
    parser.add_argument("--w-div", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--skip-plot", action="store_true", help="Skip auto-plot generation after evaluation")
    parser.add_argument("--plot-path", type=str, default="", help="Optional explicit output path for generated plot")

    config = parser.parse_args()

    if not config.train_eval and not config.eval_only and not config.plot_only:
        config.train_eval = True

    return config
