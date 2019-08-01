#!/usr/bin/env python
#importing libraries
from mpi4py import MPI
import numpy as np
from random import random
import time
import sys

#initializing important variables
comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()
N= int(sys.argv[1])

init = MPI.Wtime()

if worker !=0:

    #receiving data from master
    data = comm.recv()
    V1_parallel = data['V1']
    V2_parallel = data['V2']

    #adding received vectors and sendind back to master
    comm.send({'SUM' : V1_parallel+ V2_parallel,
               'owner' : worker}, dest=0)

else:

    #vector of random numbers between 0 and 10
    V1 = np.random.randint(1,10,N).astype(int)
    V2 = np.random.randint(1,10,N).astype(int)
    V3 = np.zeros(N)

    #fraction of data for each worker
    data_frac = int(N/(num_workers))

    i=0
    #sending data to workers
    for i in range(1,num_workers):
        data  = {'V1': V1[data_frac*(i-1):data_frac*i],
                 'V2': V2[data_frac*(i-1):data_frac*i]}
        comm.send(data, dest=i)

    #data that the master keeps
    V1_parallel = V1[data_frac*i:]
    V2_parallel = V2[data_frac*i:]

    V3[data_frac*i:] = V1_parallel + V2_parallel

    #receiving and merging information from workers
    for i in range(1,num_workers):
        data = comm.recv()
        j = data['owner']
        V3[data_frac*(j-1):data_frac*j]=data['SUM']

    elapsed = MPI.Wtime()-init

    print("Final Vector:", V3)
    print("Elapsed:", elapsed)
    print("Difference between distributed and sequential:", np.sum(V3-(V1+V2)))

