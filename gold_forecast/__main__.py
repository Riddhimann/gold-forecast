"""Command line entry point: python -m gold_forecast run."""

import argparse

from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Rebuild the gold forecasting study and reports.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("run", help="Run all experiments, tests of performance, and figures")
    command.add_argument("--data", default="main.csv")
    command.add_argument("--output", default="artifacts")
    command.add_argument("--quick", action="store_true", help="Smaller tuning grids; same honest evaluation")
    command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data, args.output, args.quick, args.seed)


if __name__ == "__main__":
    main()
