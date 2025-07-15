# !pip install Nevergrad
import os
import sys
import time
import argparse
import shlex
import subprocess
import nevergrad as ng
import joblib
import logging
from datetime import datetime
from shutil import rmtree

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from ioflexpredict import ioflexpredict
from ioflexheader import (
    get_config_map,
    set_hints_with_ioflex,
    set_hints_env_romio,
    set_hints_env_cray,
    are_cray_hints_valid,
)
from ioflexsetstriping import setstriping
from utils import header


def get_optimizer(optimizer_name, params=None, max_trials=50):

    match optimizer_name:
        case "ngioh":
            optimizer_desc = "NgIohTuned"
            optimizer_class = ng.optimizers.NgIohTuned(
                parametrization=params, budget=max_trials
            )
        case "twopde":
            optimizer_desc = "Two Points Differential Evolution"
            optimizer_class = ng.optimizers.TwoPointsDE(
                parametrization=params, budget=max_trials
            )
        case "pdopo":
            optimizer_desc = "PortfolioDiscreteOnePlusOne"
            optimizer_class = ng.optimizers.PortfolioDiscreteOnePlusOne(
                parametrization=params, budget=max_trials
            )
        case "tbpsa":
            optimizer_desc = "TBPSA Test-based population-size adaptation"
            optimizer_class = ng.optimizers.TBPSA(
                parametrization=params, budget=max_trials
            )
        case _:
            optimizer_desc = "NGOpt"
            optimizer_class = ng.optimizers.NGOpt(
                parametrization=params, budget=max_trials
            )

    logging.info(f"Running with {optimizer_desc}")

    return optimizer_class


# Application Specific
files_to_stripe = []
files_to_clean = []

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def eval_func(**kwargs):

    dir_path = os.environ.get("PWD", os.getcwd())

    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")

    sample_instance = {}
    for key in config_space.keys():
        sample_instance[key] = kwargs[key]

    # if hints == "cray":
    # if options are not valid skip the trial
    # if not are_cray_hints_valid(sample_instance, num_ranks, num_nodes):
    # raise TrialPruned()

    if ioflexset:
        set_hints_with_ioflex(sample_instance, config_path)
        os.environ["IOFLEX_HINTS"] = config_path
    else:
        if hints == "romio":
            set_hints_env_romio(sample_instance, config_path)
            os.environ["ROMIO_HINTS"] = config_path
        if hints == "cray":
            crayhints = set_hints_env_cray(sample_instance, config_path)
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


def ioflexnevergrad():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ioflex", action="store_true", default=False, help="Enable IOFlex"
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="./OutIOFlexNevergrad.csv",
        help="Path to output CSV file",
    )
    ap.add_argument(
        "--inng", "-i", type=str, default=None, help="Nevergrad study pickled file"
    )
    ap.add_argument(
        "--outng",
        "-o",
        type=str,
        default="./ng.pkl",
        help="Path to Nevergrad Study object",
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
        "--optimizer",
        type=str,
        choices=["ngioh", "twopde", "pdopo", "tbpsa", "ngopt"],
        default="ngopt",
        help="Nevergrad Optimizer Algorithm",
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

    global num_ranks, num_nodes
    num_ranks = args["num_ranks"]
    num_nodes = args["num_nodes"]

    max_trials = args["max_trials"]

    opt_name = args["optimizer"]

    global model
    model = joblib.load(args["with_model"]) if args["with_model"] else None

    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    global hints, config_space
    # Define configurations mappings
    hints = args["with_hints"]

    CONFIG_MAP = get_config_map(hints)
    config_space = {key: value for key, value in sorted(CONFIG_MAP.items()) if value}

    params = ng.p.Instrumentation(
        **{
            key: (
                ng.p.TransitionChoice(value)
                if all(isinstance(v, int) for v in value)
                else ng.p.Choice(value)
            )
            for key, value in sorted(CONFIG_MAP.items())
            if value
        }
    )
    header_items = list(config_space.keys())

    header_items.append("elapsedtime")
    if args["with_model"]:
        header_items.append("predicttime")

    outfile.write(",".join(header_items) + "\n")

    ngoptimizer = get_optimizer(opt_name, params, max_trials)
    # logger.info(f"Best trial: {study.best_trial}")
    #    joblib.dump(study, args["outoptuna"])
    recommendation = ngoptimizer.minimize(eval_func)

    print("✅ Best configuration found:")
    print(recommendation.kwargs)
    outfile.close()
    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    header.printheader()
    ioflexnevergrad()
