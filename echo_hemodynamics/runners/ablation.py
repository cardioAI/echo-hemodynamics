"""CLI entrypoint for the ablation study: ``python -m echo_hemodynamics.runners.ablation``."""

from ..ablation.study import run_ablation_study


def main():
    run_ablation_study()


if __name__ == "__main__":
    main()
