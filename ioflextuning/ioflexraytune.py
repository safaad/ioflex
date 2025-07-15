# !pip install Nevergrad
import os
import sys
import time
import argparse
import shlex
import subprocess
import nevergrad as ng
import joblib
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import optuna
from ioflexheader import SAMPLER_MAP
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.zoopt import ZOOptSearch
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


def get_optimizer(optimizer_name):

    match optimizer_name:
        case "ngioh":
            optimizer_desc = "NgIohTuned"
            optimizer_class = ng.optimizers.NgIoh
        case "twopde":
            optimizer_desc = "Two Points Differential Evolution"
            optimizer_class = ng.optimizers.TwoPointsDE
        case "pdopo":
            optimizer_desc = "PortfolioDiscreteOnePlusOne"
            optimizer_class = ng.optimizers.PortfolioDiscreteOnePlusOne
        case "tbpsa":
            optimizer_desc = "TBPSA Test-based population-size adaptation"
            optimizer_class = ng.optimizers.TBPSA
        case _:
            optimizer_desc = "NGOpt"
            optimizer_class = ng.optimizers.NGOpt

    print("Running with Optimizer " + optimizer_desc + "\n")
    return optimizer_class


def get_algorithm(algo_name, opt_name=None):

    match algo_name:
        case "optuna":
            sampler_desc, sampler_class = SAMPLER_MAP[opt_name]
            print("Using sampler: ", sampler_desc) 
            search_algo = OptunaSearch(sampler=sampler_class())
        case "ax":
            search_algo = AxSearch()
        case "hyperopt":
            search_algo = HyperOptSearch()
        case "bohb":
            search_algo = TuneBOHB()
        case "nevergrad":
            opt = get_optimizer(opt_name)
            search_algo = NevergradSearch(
                optimizer=opt, metric="elapsedtime", mode="min"
            )
        case "zoopt":
            search_algo = ZOOptSearch(budget=max_trials)

    return search_algo


# Application Specific
files_to_stripe = []
files_to_clean = []


def eval_func(sample_instance):

    dir_path = os.environ.get("PWD", os.getcwd())

    config_path = os.path.join(dir_path, "config.conf" if ioflexset else "romio-hints")

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
        predtime.append(ioflexpredict.predict_instance(model, sample_instance))

    if logisset:
        with open(logfile_o, "w") as logfile:
            logfile.write(f"Config: {configs_str}\n\n{out.decode()}\n")
        with open(logfile_e, "w") as logfile:
            logfile.write(f"Config: {configs_str}\n\n{err.decode()}\n")

    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f) if os.path.isfile(f) else rmtree(f)

    tune.report({"elapsedtime": elapsedtime})


def ioflexraytune():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ioflex", action="store_true", default=False, help="Enable IOFlex"
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default="./RayResults.csv",
        help="Path to Results CSV file",
    )
    ap.add_argument(
        "--outray",
        "-r",
        type=str,
        default="./ray_results",
        help="Path to RayTune results dir",
    )
    ap.add_argument(
        "--prev",
        "-p",
        type=str,
        default=None,
        help="Path to the directory with previous ray experiments",
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
        "--tuner",
        "-t",
        type=str,
        choices=["optuna", "ax", "bohb", "hyperopt", "nevergrad", "zoopt"],
        default="optuna",
        help="Ray Tune Search Algorithms",
    )
    ap.add_argument(
        "--optimizer",
        type=str,
        choices=["ngioh", "twopde", "pdopo", "tbpsa", "ngopt"],
        default="ngopt",
        help="Nevergrad optimizer algorithm, only valid if nevergrad is enabled",
    )
    ap.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "rand", "gp", "nsga", "brute", "grid", "auto"],
        default="tpe",
        help="Optuna samplers, only valid if optuna is enabled",
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

    args = ap.parse_args()
    args_dict = vars(args)
    
    # Example usage
    print("Selected algorithm:", args_dict["tuner"])
    args = vars(ap.parse_args())

    global ioflexset, run_app, outfile, logisset, logfile_o, logfile_e
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = args["outfile"]

    global num_ranks, num_nodes, model, hints, predtime, max_trials
    predtime = []
    num_ranks = args["num_ranks"]
    num_nodes = args["num_nodes"]
    max_trials = args["max_trials"]

    model = joblib.load(args["with_model"]) if args["with_model"] else None
    logisset = bool(args["with_log_path"])

    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = os.path.join(args["with_log_path"], f"out.{timestamp}")
        logfile_e = os.path.join(args["with_log_path"], f"err.{timestamp}")

    hints = args["with_hints"]

    CONFIG_MAP = get_config_map(hints)
    params = {
        key: (tune.choice(value)) for key, value in sorted(CONFIG_MAP.items()) if value
    }

    tuner = args["tuner"]
    match tuner:
        case "nevergrad":
            algo = args["optimizer"]
        case _:
            algo = args["sampler"]

    search_algo = get_algorithm(tuner, algo)
    storage_uri = f"file://{os.path.abspath(args["outray"])}"
    if args["prev"] is not None:
        search_algo.restore_from_dir(os.path.join(args["prev"], "raytune_io"))
        tuner = tune.Tuner(
            eval_func,
            tune_config=tune.TuneConfig(
                search_alg=search_algo,
                num_samples=max_trials,
                metric="elapsedtime",
                mode="min",
            ),
            run_config=tune.RunConfig(
                storage_path=storage_uri, name="raytune_io", log_to_file=True
            ),
        )
    else:
        tuner = tune.Tuner(
            eval_func,
            tune_config=tune.TuneConfig(
                search_alg=search_algo,
                num_samples=max_trials,
                metric="elapsedtime",
                mode="min",
            ),
            run_config=tune.RunConfig(
                storage_path=storage_uri, name="raytune_io", log_to_file=True
            ),
            param_space=params,
        )

    results = tuner.fit()

    prefix = "config/"
    cols_name = [prefix + c for c in params.keys()]
    cols_name.append("elapsedtime")

    df = results.get_dataframe()
    df = df[cols_name]

    if model is not None:
        df["predicttime"] = predtime
    df.to_csv(outfile, index=False)

    print(results.get_best_result(metric="elapsedtime", mode="min").config)

    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    header.printheader()
    ioflexraytune()
