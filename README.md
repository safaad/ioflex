# IOFlex 
IOFlex is a portable I/O wrapper tool that transperantly sets I/O configurations for HPC applications. IOFlex can be used to optimize IO for HPC applications without changing the source code. It overrides hints set using environment variables and MPI_Info_set().

## Instructions
### Prerequisites

- libconfuse: https://github.com/libconfuse/libconfuse.git

### Building and Installation

```bash
$ ./autogen.sh
# By default, ioflex is installed in `pwd` if not changed via --prefix
$ ./configure --with-libconfuse=<path to libconfuse> --prefix=<path to install dir>
$ make && make install
```

### Setting IO Configurations

- ```config.conf``` file by default specifies the IO configs IOFlex should adapt. 
- Supported IO configurations
``` 
striping_unit = ...
striping_factor = ...
romio_filesystem_type = ...
romio_ds_write = ...
romio_ds_read = ...
romio_cb_read = ...
romio_cb_write = ...
cb_nodes = ...
cb_buffer_size = ...
cb_config_list = ...
```

- Parameters absent from the configuration file are left unset, retaining the values previously set by the application or default values if none were specified.

### Using IOFlex shared library
- Using LD_PRELOAD without changing source code:
```bash
export LD_PRELOAD=<path-to-libioflex>/libioflex.so
mpirun ...
```
- OR, by recompiling with shared library flags
```
-L$IOFLEX_LIB_PATH -lioflex -Wl,-rpath=$IOFLEX_LIB_PATH

```
### Important Notes
- You can use Darshan I/O profiler with IOFlex. Just make sure to load IOFlex before Darshan
- Compile and install IOFlex with the same environment (compiler and MPI) the program use
- Running with MPT:
    - Set the following flag ```export MPI_SHEPHERD=true```
    - For MPT + OpenMP: LD_PRELOAD hangs so recompile with shared library flags
- Running with OpenMPI:
    -   Use ROMIO MCA component ```mpirun -np $NP --mca io romio321 <exe>```
   

# IOFlex GA



