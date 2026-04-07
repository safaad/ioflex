import json
import sys

# Script to generate  cray-hints string given a json file
def generate_crayhints_cmd(params_file):
    with open(params_file, 'r') as f:
        data = json.load(f)
    
    hints = [f"{key}={value}" for key, value in data.items()]
    hint_string = "*:" + ":".join(hints)

    return hint_string


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_hints.py <params_file.json>")
        sys.exit(1)

    params_file = sys.argv[1]
    result = generate_crayhints_cmd(params_file)

    print(f"export MPICH_MPIIO_HINTS={result}")