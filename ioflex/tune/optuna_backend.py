# !pip install optuna
# Best Samplers for Optuna  Categorial RandSampler, GridSampler, TPESampler, NSGAIISampler, GPSampler, BruteForceSampler, Simulated Annealing
# GP requires torch
import os
import time
import argparse
import shlex
import subprocess
import joblib
import logging
from datetime import datetime
import optuna
from ioflex.model import base
from ioflex.common import SAMPLER_MAP, PRUNER_MAP
from ioflex.common import (
    get_config_map,
    set_hints_with_ioflex,
    set_hints_env_romio,
    set_hints_env_cray,
    are_cray_hints_valid,
    get_bandwidth_darshan,
    remove_path,
)
from ioflex.striping import setstriping
from optuna.exceptions import TrialPruned

# Application Specific
files_to_stripe = []
files_to_clean = []


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


# Default ROMIO Configuration
def eval_func(
    trial,
    config=get_config_map("romio"),
    model=None,
):

    dir_path = os.environ.get("PWD", os.getcwd())
    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")

    sample_instance = {}
    for key, values in config.items():
        if not values:
            continue
        selected_val = trial.suggest_categorical(key, values)
        sample_instance[key] = selected_val

    if hints == "cray":
        # if options are not valid skip the trial
        if not are_cray_hints_valid(sample_instance, num_ranks, num_nodes):

            raise TrialPruned()

    if ioflexset:
        set_hints_with_ioflex(sample_instance, config_path)
        os.environ["IOFLEX_HINTS"] = config_path
    else:
        if hints == "romio":
            set_hints_env_romio(sample_instance, config_path)
            os.environ["ROMIO_HINTS"] = config_path
        if hints == "cray":
            crayhints = set_hints_env_cray(sample_instance)
            os.environ["MPICH_MPIIO_HINTS"] = crayhints
        # if hints == "ompio":
        #  TODO

    configs_str = ",".join(map(str, sample_instance.values()))

    stripe_count = (
        int(sample_instance["striping_factor"])
        if "striping_factor" in sample_instance
        else 8
    )
    stripe_size = (
        str(sample_instance["striping_unit"] // 1048576) + "M"
        if "striping_unit" in sample_instance
        else "1M"
    )
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
    objective = time.time() - start_time

    outline = f"{configs_str},{objective}"
    if tune_bandwidth:
        darshan_dir = os.environ["DARSHAN_LOG_DIR_PATH"]
        log_path = os.path.join(darshan_dir, "*.darshan")
        # get MPI-IO bandwidth MiB/s
        objective = get_bandwidth_darshan(log_path, "MPI-IO")
        if objective == -1:
            raise TrialPruned()
        outline = f"{outline},{objective}"
    if model:
        pred = base.predict_instance(model, sample_instance)
        outline = f"{outline},{pred}\n"
    else:
        outline = f"{outline}\n"

    outfile.write(outline)

    if logisset:
        logfile_o.write(f"Config: {outline}\n\n{out.decode()}\n")
        logfile_e.write(f"Config: {outline}\n\n{err.decode()}\n")

    for f in files_to_clean:
        remove_path(f)

    print(f"Running config: {outline}")

    return objective


def run(args=None):

    ap = argparse.ArgumentParser(prog="ioflex tune --optuna")
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
        "--num_ranks",
        "-np",
        type=int,
        required=True,
        help="Number of ranks used to run the program",
    )
    ap.add_argument(
        "--num_nodes", "-n", type=int, required=True, help="Number of nodes allocated"
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
    ap.add_argument(
        "-b",
        "--tune_bandwidth",
        action="store_true",
        default=False,
        help="Enable IOFlex",
    )
    args = vars(ap.parse_args(args))

    global ioflexset, run_app, outfile, logisset, logfile_o, logfile_e, hints, tune_bandwidth
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    tune_bandwidth = args["tune_bandwidth"]
    outfile = open(args["outfile"], "w")

    global num_ranks, num_nodes
    num_ranks = args["num_ranks"]
    num_nodes = args["num_nodes"]

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
    config_space = {key: value for key, value in sorted(CONFIG_MAP.items()) if value}
    header_items = list(config_space.keys())

    header_items.append("elapsedtime")
    if args["tune_bandwidth"]:
        header_items.append("I/O-Bandwidth-Mib/s")

    if args["with_model"]:
        header_items.append("Predicted-Objective")

    outfile.write(",".join(header_items) + "\n")

    sampler = get_sampler(args["sampler"], config_space)
    pruner = get_pruner(args["pruner"])

    if args["inoptuna"]:
        study = joblib.load(args["inoptuna"])
    else:
        direction = "minimize" if not tune_bandwidth else "maximize"
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name="ioflexoptuna_study",
        )

    study.optimize(
        lambda trial: eval_func(trial, config_space, model), n_trials=args["max_trials"]
    )

    print(f"Best trial: {study.best_trial}")
    joblib.dump(study, args["outoptuna"])

    outfile.close()
    if logisset:
        logfile_o.close()
        logfile_e.close()
