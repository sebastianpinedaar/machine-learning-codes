#Author: Sebastian Pineda Arango
#Institution: University of Hildesheim
#Year: 2019
#Functionality: This script should be called with mpiexec and is
#able to find the average of a randomly generated using several processes
#that communicate with each other using MPI.
# #importing libraries
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
    V_parallel = data['V']
    sum_V = sum(V_parallel)
    comm.send({'SUM':sum_V,
               'owner': worker}, dest=0)

else:
    #vector of random numbers between 1 and 10
    V = np.random.randint(2,5,N).astype(int)

    #fraction of data for each worker
    data_frac = int(N/(num_workers))

    i=0 #to avoid crashing when P=1

    #sending data to workers
    for i in range(1,num_workers):
        data  = {'V': V[data_frac*(i-1):data_frac*i]}
        comm.send(data, dest=i)

    #sum of vector part in master
    sum_V = sum(V[data_frac*i:])

    #receiving sums from workers and adding to the accumulator
    for i in range(1,num_workers):
        data = comm.recv()
        sum_V += data['SUM']

    print("Total agerage:", sum_V/N)
    
    elapsed = MPI.Wtime()-init
    print("Elapsed:", elapsed)
    print("Difference between distributed and sequential:", (sum_V/N)-np.mean(V))



