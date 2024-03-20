# IO Flex Wrapper Library 
## pre-requisites
Libconfuse

https://github.com/libconfuse/libconfuse.git


### MPT
```export MPI_SHEPHERD=true```

### OpenMPI
```mpirun -np $NP --mca io romio321 <exe>```
