# IOFlex

IOFlex is a portable I/O tuning framework designed to efficiently optimize I/O performance and automatically tune I/O configurations for HPC (High-Performance Computing) applications.

IOFlex Consists of two main component (i) IOFlex I/O wrapper library and (ii) IOFlex I/O Tune library.

## Table of Contents

- [Features](#features)
- [IOFLex I/O Wrapper](#ioflex-io-wrapper)
- [IOFlex Tune Library](#ioflex-tune-library)

## Features

- An MPI-I/O wrapper library to adjust I/O hints without requiring source code modifications.
- It supports multiple MPI implementations and higher level I/O libraries such as HDF5 and PNetCDF.
- It works with profiling tools like Darshan.
- A python library to tune I/O configurations efficiently. IOFlex tune integrates various state-of-the-art hyperparameter tuning libraries including Ray Tune, Nevergrad, and Optuna.
- Flexible and portable library.

## IOFlex I/O Wrapper

IOFlex adjusts I/O configurations including Lustre striping and MPI-I/O hints. It integrates seamlessly, either through LD_PRELOAD at runtime or by linking the shared library at compilation. Its I/O wrapper intercepts MPI-I/O calls and applies the specified I/O hints.

IOFlex supports various MPI environments, including OpenMPI, Cray-MPICH and ROMIO, and is compatible with profiling tools like Darshan. Any I/O configurations set through IOFlex override previous settings.

By intercepting applications at the MPI-I/O layer, IOFlex also supports higher level I/O libraries such as HDF5 and PNetCDF.

### Prerequisites

- libconfuse: https://github.com/libconfuse/libconfuse.git

### Building and Installation

```bash
$ ./autogen.sh
# By default, ioflex is installed in `pwd` if not changed via --prefix
$ ./configure --with-libconfuse=<path-to-libconfuse> --with-hints=[romio|ompio|cray] --prefix=<path-to-install-dir>
$ make && make install
```

### Setting IO Configurations

- Define the path to the I/O configuration file by setting the environment variable `IOFLEX_HINTS`. Check `config.conf` file to see an example of how to set the I/O configurations.

- Parameters absent from the configuration file are left unset, retaining the values previously set by the application or default values if none were specified.

- Supported IO configurations and implementations

|                                                                                                           **ROMIO I/O Hints**                                                                                                           |                                                                                                                                   **Cray-MPICH I/O Hints**                                                                                                                                   |     **OMPIO**     |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------: |
| `striping_unit` <br/> `striping_factor` <br/> `romio_filesystem_type` <br/> `romio_ds_read` <br/> `romio_ds_write` <br/> `romio_cb_read` <br/> `romio_cb_write` <br/> `cb_nodes` <br/> `cb_config_list` <br/> `romio_no_indep_rw` <br/> | `striping_unit` <br/> `striping_factor` <br/> `romio_ds_read` <br/> `romio_ds_write` <br/> `romio_cb_read` <br/> `romio_cb_write` <br/> `cb_nodes` <br/> `cb_config_list` <br/> `romio_no_indep_rw` <br/> `cray_cb_nodes_multiplier` <br/> `cray_cb_write_lock_mode` <br/> `direct_io` <br/> | To be implemented |

### Usage

#### 1. Using `LD_PRELOAD` (no recompilation)

```bash
export LD_PRELOAD=<path-to-libioflex>/libioflex.so
mpirun -np <N> ./my_app
```

#### 2. Linking explicitly (requries recompilation)

```bash
mpicc my_app.c -L$IOFLEX_LIB_PATH -lioflex -Wl,-rpath=$IOFLEX_LIB_PATH

```

### Important Notes

- You can use Darshan I/O profiler with IOFlex. Just make sure to load IOFlex before Darshan
- IOFlex supports HDF5 and NetCDF I/O libraries.
- Compile and install IOFlex with the same environment (compiler and MPI) the program use
- Running with MPT:
  - Set the following flag `export MPI_SHEPHERD=true`
  - For MPT + OpenMP: LD_PRELOAD hangs so recompile with shared library flags
- Running with OpenMPI:
  - Use ROMIO MCA component `mpirun -np $NP --mca io romio321 <exe>`

## IOFlex Tune Library

- I/O on HPC systems involves multiple software and hardware layers, each with tunable parameters. Default configurations are often far from optimal, and require tuning. However, the complexity of the I/O stack combined with the diversity of access patterns makes manual parameter tuning time-consuming and labor-intensive.

- The IOFlex Tune library leverages advanced search algorithms to efficiently explore the I/O hyperparameter space, identifying near-optimal configurations without requiring exhaustive searches.

### Installation

To install IOFlex Tune and its dependencies run

```bash
pip -e install .
```

### IOFlex -- Usage of Tuning Backends

IOFlex can be invoked as a module. The `tune` subcommand accepts flags to which backend to use. The backends include Optuna `--optuna`, Ray Tune `--ray`, and Nevergrad `--nevergrad`. Each backend has its own options.

```bash
python -m ioflex tune [BACKEND] [options...]
```

#### 1. Optuna backend

Run a tuning study with Optuna:

```bash
usage: ioflex tune --optuna [-h] [--ioflex] [--outfile OUTFILE] [--inoptuna INOPTUNA]
                            [--outoptuna OUTOPTUNA] --num_ranks NUM_RANKS --num_nodes NUM_NODES --cmd
                            [CMD ...] [--max_trials MAX_TRIALS]
                            [--sampler {tpe,rand,gp,nsga,brute,grid,auto}]
                            [--pruner {hyper,median,successivehalving,nop}] [--with_log_path WITH_LOG_PATH]
                            [--with_model WITH_MODEL] [--with_hints {romio,cray,ompio}] [-b] [--config CONFIG]
```

Options:

```
  -h, --help            show this help message and exit
  --ioflex              Enable IOFlex
  --outfile OUTFILE     Path to output CSV file
  --inoptuna INOPTUNA   Optuna study pickled file
  --outoptuna OUTOPTUNA, -o OUTOPTUNA
                        Path to Optuna Study output
  --num_ranks NUM_RANKS, -np NUM_RANKS
                        Number of ranks used to run the program
  --num_nodes NUM_NODES, -n NUM_NODES
                        Number of nodes allocated
  --cmd [CMD ...], -c [CMD ...]
                        Application command line
  --max_trials MAX_TRIALS
                        Max number of trials
  --sampler {tpe,rand,gp,nsga,brute,grid,auto}
                        Optuna sampler
  --pruner {hyper,median,successivehalving,nop}
                        Optuna pruner
  --with_log_path WITH_LOG_PATH
                        Output logging path
  --with_model WITH_MODEL
                        Path to trained prediction model
  --with_hints {romio,cray,ompio}
                        MPIIO hints mode
  -b, --tune_bandwidth  Use I/O bandwidth as the tuning objective
  --config CONFIG       Path to JSON configuration file (default: ../configs/tune_config_romio.json
```

#### 2. Ray Tune Backend

```bash
usage: ioflex tune --ray [-h] [--ioflex] [--outfile OUTFILE] [--outray OUTRAY] [--prev PREV] --num_ranks
                   NUM_RANKS --num_nodes NUM_NODES --cmd [CMD ...] [--max_trials MAX_TRIALS]
                   [--tuner {optuna,ax,bohb,hyperopt,nevergrad,zoopt}]
                   [--optimizer {ngioh,twopde,pdopo,tbpsa,ngopt}]
                   [--sampler {tpe,rand,gp,nsga,brute,grid,auto}] [--with_log_path WITH_LOG_PATH]
                   [--with_model WITH_MODEL] [--with_hints {romio,cray,ompio}] [-b] [--config CONFIG]
```

Options:

```
options:
  -h, --help            show this help message and exit
  --ioflex              Enable IOFlex
  --outfile OUTFILE     Path to Results CSV file
  --outray OUTRAY, -r OUTRAY
                        Path to RayTune results dir
  --prev PREV, -p PREV  Path to the directory with previous ray experiments
  --num_ranks NUM_RANKS, -np NUM_RANKS
                        Number of ranks used to run the program
  --num_nodes NUM_NODES, -n NUM_NODES
                        Number of nodes allocated
  --cmd [CMD ...], -c [CMD ...]
                        Application command line
  --max_trials MAX_TRIALS
                        Max number of trials
  --tuner {optuna,ax,bohb,hyperopt,nevergrad,zoopt}, -t {optuna,ax,bohb,hyperopt,nevergrad,zoopt}
                        Ray Tune Search Algorithms
  --optimizer {ngioh,twopde,pdopo,tbpsa,ngopt}
                        Nevergrad optimizer algorithm, only valid if nevergrad is enabled
  --sampler {tpe,rand,gp,nsga,brute,grid,auto}
                        Optuna samplers, only valid if optuna is enabled
  --with_log_path WITH_LOG_PATH
                        Output logging path
  --with_model WITH_MODEL
                        Path to trained prediction model
  --with_hints {romio,cray,ompio}
                        MPIIO hints mode
  -b, --tune_bandwidth  Use I/O bandwidth as the tuning objective
  --config CONFIG       Path to JSON configuration file (default: ../configs tune_config_romio.json
```

### 3. Nevergrad

```bash
usage: ioflex tune --nevergrad [-h] [--ioflex] [--outfile OUTFILE] [--inng INNG] [--outng OUTNG] --num_ranks NUM_RANKS
                   --num_nodes NUM_NODES --cmd [CMD ...] [--max_trials MAX_TRIALS]
                   [--optimizer {ngioh,twopde,pdopo,tbpsa,ngopt}] [--with_log_path WITH_LOG_PATH]
                   [--with_model WITH_MODEL] [--with_hints {romio,cray,ompio}] [-b] [--config CONFIG]
```

Options

```
  -h, --help            show this help message and exit
  --ioflex              Enable IOFlex
  --outfile OUTFILE     Path to output CSV file
  --inng INNG, -i INNG  Nevergrad study pickled file
  --outng OUTNG, -o OUTNG
                        Path to Nevergrad Study object
  --num_ranks NUM_RANKS, -np NUM_RANKS
                        Number of ranks used to run the program
  --num_nodes NUM_NODES, -n NUM_NODES
                        Number of nodes allocated
  --cmd [CMD ...], -c [CMD ...]
                        Application command line
  --max_trials MAX_TRIALS
                        Max number of trials
  --optimizer {ngioh,twopde,pdopo,tbpsa,ngopt}
                        Nevergrad Optimizer Algorithm
  --with_log_path WITH_LOG_PATH
                        Output logging path
  --with_model WITH_MODEL
                        Path to trained prediction model
  --with_hints {romio,cray,ompio}
                        MPIIO hints mode
  -b, --tune_bandwidth  Use I/O bandwidth as the tuning objective
  --config CONFIG       Path to JSON configuration file (default: ../configs/tune_config_romio.json
```
