# ioflex/__main__.py
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ioflex <command> [args...]")
        sys.exit(1)

    command = sys.argv[1]
    from .utils import header
    header.printheader()
    if command == "tune":
        if "--optuna" in sys.argv:
            from .tune import optuna_backend
            optuna_args = sys.argv[sys.argv.index("--optuna") + 1 :]
            optuna_backend.run(optuna_args)
        elif "--nevergrad" in sys.argv:
            from .tune import nevergrad_backend
            nevergrad_args = sys.argv[sys.argv.index("--nevergrad") + 1 :]
            nevergrad_backend.run(nevergrad_args)
        else:
            print("Please specify a backend: --optuna or --nevergrad")
    elif command == "model":

        if "--sample" in sys.argv:
            from .model import sampler
            sampler_args = sys.argv[sys.argv.index("--sample") + 1 :]
            sampler.run(sampler_args)
        elif "--rank" in sys.argv:
            from .model import rank
            rank_args = sys.argv[sys.argv.index("--rank") + 1 :]
            rank.run(rank_args)
        else:
           print("Please specify a subcommand: --sample or --rank") 
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
