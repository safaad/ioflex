# !pip install optuna
# Best Samplers for Optuna  Categorial RandSampler, GridSampler, TPESampler, NSGAIISampler, GPSampler, BruteForceSampler, Simulated Annealing
# GP requires torch
import os
import sys
import time
import argparse
import shlex
import subprocess
import optuna
import joblib
import logging
from datetime import datetime
from shutil import rmtree

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from ioflexpredict import ioflexpredict
from ioflexheader import SAMPLER_MAP, PRUNER_MAP, CONFIG_ROMIO_MAP, get_config_map
from ioflexsetstriping import setstriping
from utils import header


# Application Specific
files_to_stripe = []
files_to_clean = []

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_sampler(sampler_name, config_space=None):

    if sampler_name not in SAMPLER_MAP:
        raise ValueError(
            f"Invalid sampler: '{sampler_name}'. Choose from {list(SAMPLER_MAP.keys())}."
        )

    sampler_desc, sampler_class = SAMPLER_MAP[sampler_name]
    logging.info(f"Running with {sampler_desc}")

    return sampler_class() if sampler_name != "grid" else sampler_class(config_space)


def get_pruner(pruner_name):

    if pruner_name not in PRUNER_MAP:
        logging.warning(f"Invalid pruner '{pruner_name}'. Using NopPruner by default.")
        return optuna.pruners.NopPruner()

    logging.info(f"Using {pruner_name} pruner.")
    return PRUNER_MAP[pruner_name]()


def eval_func(
    trial,
    config=CONFIG_ROMIO_MAP,
    model=None,
):

    dir_path = os.environ.get("PWD", os.getcwd())

    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")

    stripe_count = 8
    stripe_size = "1M"
    with open(config_path, "w") as config_file:
        sample_instance = {}
        configs_str = ""
        configs_list = []

        for key, values in config.items():
            if not values:
                continue

            selected_val = trial.suggest_categorical(key, values)
            sample_instance[key] = selected_val
            configs_list.append(str(selected_val))

            separator = " = " if ioflexset else " "

            # Special handling for specific keys
            match key:
                case "striping_factor":
                    stripe_count = int(selected_val)
                case "striping_unit":
                    stripe_size = str(selected_val // 1048576) + "M"
                    config_file.write(f"cb_buffer_size{separator}{selected_val}\n")
                case "romio_filesystem_type":
                    os.environ.update({"ROMIO_FSTYPE_FORCE": selected_val})
                case "cb_config_list":
                    if ioflexset:
                        config_file.write(f'{key}{separator}"{selected_val}"\n')
                        continue
            config_file.write(f"{key}{separator}{selected_val}\n")

    configs_str = ",".join(configs_list)
    # Use ROMIO HINTS if IOFLex is not enabled
    if not ioflexset:
        os.environ["ROMIO_HINTS"] = config_path
    else:
        os.environ["IOFLEX_HINTS"] = config_path

    # Set striping with lfs setstripe
    for f in files_to_stripe:
        setstriping(f, stripe_count, stripe_size)

    start_time = time.time()
    process = subprocess.Popen(
        shlex.split(run_app),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        cwd=os.getcwd(),
    )
    out, err = process.communicate()
    elapsedtime = time.time() - start_time

    if model:
        predtime = ioflexpredict.predict_instance(model, sample_instance)
        outline = f"{configs_str},{elapsedtime},{predtime}\n"
    else:
        outline = f"{configs_str},{elapsedtime}\n"

    outfile.write(outline)

    if logisset:
        logfile_o.write(f"Config: {configs_str}\n\n{out.decode()}\n")
        logfile_e.write(f"Config: {configs_str}\n\n{err.decode()}\n")

    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f) if os.path.isfile(f) else rmtree(f)

    logger.info(f"Running config: {outline}")

    return elapsedtime


def ioflexoptuna():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ioflex", action="store_true", default=False, help="Enable IOFlex"
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="./OutIOFlexOptuna.csv",
        help="Path to output CSV file",
    )
    ap.add_argument(
        "--inoptuna", type=str, default=None, help="Optuna study pickled file"
    )
    ap.add_argument(
        "--outoptuna",
        "-o",
        type=str,
        default="./optuna_study.pkl",
        help="Path to Optuna Study output",
    )
    ap.add_argument(
        "--cmd",
        "-c",
        type=str,
        required=True,
        nargs="*",
        help="Application command line",
    )
    ap.add_argument("--max_trials", type=int, default=50, help="Max number of trials")
    ap.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "rand", "gp", "nsga", "brute", "grid", "auto"],
        default="tpe",
        help="Optuna sampler",
    )
    ap.add_argument(
        "--pruner",
        type=str,
        choices=["hyper", "median", "successivehalving", "nop"],
        default="nop",
        help="Optuna pruner",
    )
    ap.add_argument(
        "--with_log_path", type=str, default=None, help="Output logging path"
    )
    ap.add_argument(
        "--with_model", type=str, default=None, help="Path to trained prediction model"
    )
    ap.add_argument(
        "--with_hints",
        type=str,
        default="romio",
        help="MPIIO hints mode",
        choices=["romio", "cray", "ompio"],
    )
    args = vars(ap.parse_args())

    global ioflexset, run_app, outfile, logisset, logfile_o, logfile_e
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = open(args["outfile"], "w")

    model = joblib.load(args["with_model"]) if args["with_model"] else None

    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    # Define configurations mappings
    hints = args["with_hints"]
    CONFIG_MAP = get_config_map(hints)
    config_space = {}
    header_items = []
    # Populate config_space and header list
    for key, value in CONFIG_MAP.items():
        if value:
            config_space[key] = value
            header_items.append(key)

    header_items.append("elapsedtime")
    if args["with_model"]:
        header_items.append("predicttime")

    header_str = ",".join(header_items) + "\n"
    outfile.write(header_str)

    sampler = get_sampler(args["sampler"], config_space)
    pruner = get_pruner(args["pruner"])

    if args["inoptuna"]:
        study = joblib.load(args["inoptuna"])
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name="ioflexoptuna_study",
        )

    study.optimize(
        lambda trial: eval_func(trial, config_space, model), n_trials=args["max_trials"]
    )

    logger.info(f"Best trial: {study.best_trial}")
    joblib.dump(study, args["outoptuna"])

    outfile.close()
    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    header.printheader()
    ioflexoptuna()
