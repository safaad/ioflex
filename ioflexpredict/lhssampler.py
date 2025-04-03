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
import parsedarshan
import glob
from tqdm.auto import tqdm
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from ioflexheader import CONFIG_MAP
from utils import header
from collections import defaultdict

def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

logger = configure_logging()

files_to_clean = []

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


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--ioflex', action="store_true", default=False, help="Enable IOFlex")
    ap.add_argument('--outfile', type=str, default="./lhssampledruns.csv", help="Path to Output")
    ap.add_argument('--app_name', type=str, required=True, help="Application Name")
    ap.add_argument('--num_nodes', type=int, required=True, help="Number of nodes")
    ap.add_argument('--nsamples', type=int, default=50, help="Number of samples")
    ap.add_argument('--mpi_procs', type=int, required=True, help="Number of MPI processes per node")
    ap.add_argument('--darshan_path', type=str, default=None, help="Path to Darshan output")
    ap.add_argument("--with_log_path", type=str, default=None, help="Output logging path")
    ap.add_argument('--cmd', '-c', type=str, required=True, nargs="*", help="Application command line")
    
    args = vars(ap.parse_args())
    ioflexset = args["ioflex"]
    run_app = " ".join(args["cmd"])
    outfile = args["outfile"]
    
    sampled_configs = generate_lhs_samples(list(v for v in CONFIG_MAP.values() if v), args["nsamples"])
    logger.info(f"Number of samples: {len(sampled_configs)}")
    print(sampled_configs)
    logisset = bool(args["with_log_path"]) 
    if logisset:
        os.makedirs(args["with_log_path"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
        logfile_o = open(os.path.join(args["with_log_path"], f"out.{timestamp}"), "w")
        logfile_e = open(os.path.join(args["with_log_path"], f"err.{timestamp}"), "w")

    
    df = pd.DataFrame()
    
    cols = ('appname', 'num_nodes', 'mpi_procs', 'runtime')
    cols += tuple(str(key) for key in CONFIG_MAP.keys() if CONFIG_MAP[key]) 
    
    iodict = {key : [] for key in cols}
    dictdarshan = {}
    for solution in tqdm(sampled_configs):
        
        iodict['appname'].append(args['app_name'])
        iodict['num_nodes'].append(args['num_nodes'])
        iodict['mpi_procs'].append(args['mpi_procs']
                                   )
        config_path = os.path.join(os.getcwd(), "config.conf" if ioflexset else "romio-hints")
        with open(config_path, "w") as config_file:
            config_entries = []
            for key, values in CONFIG_MAP.items():

                if not values:
                    continue

                selected_val = solution[list(k for k in CONFIG_MAP.keys() if CONFIG_MAP[k]).index(key)]
                separator = " = " if ioflexset else " "
                
                config_file.write(f"{key}{separator}{selected_val}\n")
                config_entries.append(str(selected_val))
                iodict[key].append(selected_val)
                
                if key == "striping_unit":
                    config_file.write(f"cb_buffer_size{separator}{selected_val}\n")
                elif key == "romio_filesystem_type":
                    os.environ["ROMIO_FSTYPE_FORCE"] = selected_val 

        configs_str = ",".join(config_entries)
        if not ioflexset:
            os.environ["ROMIO_HINTS"] = config_path
        
        starttime = time.time()
        process = subprocess.Popen(shlex.split(run_app), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, cwd=os.getcwd())
        out, err = process.communicate()
        elapsedtime = time.time() - starttime
        iodict["runtime"].append(elapsedtime)
        
        if logisset:
            logfile_o.write(f"Config: {configs_str}\n\n{out.decode()}\n")
            logfile_e.write(f"Config: {configs_str}\n\n{err.decode()}\n")
            
        if args["darshan_path"]:
            logfile = os.path.join(args["darshan_path"], "*.darshan")
            for f in glob.glob(logfile):
                localdictdarshan = parsedarshan.parsedarshan(f)
                if not dictdarshan:
                    dictdarshan = {key : [] for key in localdictdarshan.keys()}
                    
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
        
if __name__ == '__main__':
    header.printheader()
    main()