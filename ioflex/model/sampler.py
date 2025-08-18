import os
import sys
import time
import argparse
import shlex
import subprocess
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from shutil import rmtree
from scipy.stats import qmc
import ioflex.model.parser as parser
import glob
from tqdm.auto import tqdm


from ioflex.common import get_config_map
from ioflex.striping import setstriping

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


logger = configure_logging()

files_to_clean = []
files_to_stripe = []


def generate_lhs_samples(param_grid, nsamples):
    n_dims = len(param_grid)
    lhs = qmc.LatinHypercube(d=n_dims)
    lhs_samples = lhs.random(n=nsamples)

    sampled_configs = []
    for i in range(nsamples):
        sampled_config = []
        for j, param_values in enumerate(param_grid):
            index = int(np.floor(lhs_samples[i, j] * len(param_values)))
            sampled_config.append(param_values[index])
        sampled_configs.append(sampled_config)

    return sampled_configs


def run(args=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--ioflex", action="store_true", default=False, help="Enable IOFlex"
    )
    ap.add_argument(
        "--outfile", type=str, default="./lhssampledruns.csv", help="Path to Output"
    )
    ap.add_argument("--app_name", type=str, required=True, help="Application Name")
    ap.add_argument("--num_nodes", type=int, required=True, help="Number of nodes")
    ap.add_argument("--nsamples", type=int, default=50, help="Number of samples")
    ap.add_argument(
        "--mpi_procs", type=int, required=True, help="Number of MPI processes per node"
    )
    ap.add_argument(
        "--darshan_path", type=str, default=None, help="Path to Darshan output"
    )
    ap.add_argument(
        "--with_log_path", type=str, default=None, help="Output logging path"
    )
    ap.add_argument(
        "--cmd",
        "-c",
        type=str,
        required=True,
        nargs="*",
        help="Application command line",
    )
    ap.add_argument(
        "--with_hints",
        type=str,
        default="romio",
        help="MPIIO hints mode",
        choices=["romio", "cray", "ompio"],
    )
    args = vars(ap.parse_args())
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = args["outfile"]


    hints = args["with_hints"]
    CONFIG_MAP = get_config_map(hints)
    sampled_configs = generate_lhs_samples(
        list(v for v in CONFIG_MAP.values() if v), args["nsamples"]
    )
    logger.info(f"Number of samples: {len(sampled_configs)}")
    print(sampled_configs)
    logisset = bool(args["with_log_path"])
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    df = pd.DataFrame()

    cols = ("appname", "num_nodes", "mpi_procs", "runtime")
    cols += tuple(str(key) for key in CONFIG_MAP.keys() if CONFIG_MAP[key])

    iodict = {key: [] for key in cols}
    dictdarshan = {}
    for solution in tqdm(sampled_configs):

        iodict["appname"].append(args["app_name"])
        iodict["num_nodes"].append(args["num_nodes"])
        iodict["mpi_procs"].append(args["mpi_procs"])
        config_path = os.path.join(
            os.getcwd(), "config.conf" if ioflexset else "romio-hints"
        )
        with open(config_path, "w") as config_file:
            config_entries = []
            for key, values in CONFIG_MAP.items():

                if not values:
                    continue

                selected_val = solution[
                    list(k for k in CONFIG_MAP.keys() if CONFIG_MAP[k]).index(key)
                ]
                separator = " = " if ioflexset else " "

                config_entries.append(str(selected_val))
                iodict[key].append(selected_val)

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

        configs_str = ",".join(config_entries)
        if not ioflexset:
            os.environ["ROMIO_HINTS"] = config_path
        else:
            os.environ["IOFLEX_HINTS"] = config_path

        # Set striping with lfs setstripe
        for f in files_to_stripe:
            setstriping(f, stripe_count, stripe_size)

        starttime = time.time()
        process = subprocess.Popen(
            shlex.split(run_app),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            cwd=os.getcwd(),
        )
        out, err = process.communicate()
        elapsedtime = time.time() - starttime
        iodict["runtime"].append(elapsedtime)

        if logisset:
            logfile_o.write(f"Config: {configs_str}\n\n{out.decode()}\n")
            logfile_e.write(f"Config: {configs_str}\n\n{err.decode()}\n")

        if args["darshan_path"]:
            logfile = os.path.join(args["darshan_path"], "*.darshan")
            for f in glob.glob(logfile):
                localdictdarshan = parser.parsedarshan(f)
                if not dictdarshan:
                    dictdarshan = {key: [] for key in localdictdarshan.keys()}

                for key, value in localdictdarshan.items():
                    dictdarshan[key].append(value)
                os.remove(f)

        for f in files_to_clean:
            if os.path.exists(f):
                os.remove(f) if os.path.isfile(f) else rmtree(f)

    iodict = iodict | dictdarshan
    df = pd.DataFrame.from_dict(iodict)
    df.to_csv(outfile, index=False)

    if logisset:
        logfile_o.close()
        logfile_e.close()


if __name__ == "__main__":
    run()
