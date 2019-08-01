#Author: Sebastian Pineda Arango
#Institution: University of Hildesheim
#Year: 2019
#Functionality: This script should be called with mpiexec and is
#able to multiply a matrix (A) and a vector (B) using several process
#that communicates using MPI.

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
N = int(sys.argv[1])

init = MPI.Wtime()

if worker !=0:

    #receiving data from master
    data = comm.recv()
    A_parallel = data['A']
    b_parallel = data['b']

    #multiplying partial result
    c_partial = A_parallel@b_parallel

    #sending data back to master
    comm.send({'c_partial':c_partial,
               'owner':worker}, dest=0)

else:

    #vector of random numbers between 1 and 10
    A = np.random.randint(1,10,(N,N)).astype(int)
    b = np.random.uniform(1,10,N).astype(int)
    c = np.zeros(N)

    #fraction of data for each worker
    data_frac = int(N/(num_workers))
  
    i=0 #initializing variable to avoid crashing when P=1

    #sending data to workers
    for i in range(1,num_workers):
        data  = {'A': A[data_frac*(i-1):data_frac*i,],
                 'b': b}
        comm.send(data, dest=i)

    #multiplying data fraction in master
    c[data_frac*i:] = A[data_frac*i:,]@b

    #receiving data from workers and constructing final result
    for i in range(1,num_workers):

       data = comm.recv()
       j = data['owner']
       c[data_frac*(j-1):data_frac*j] = data['c_partial']

    elapsed = MPI.Wtime()-init


    print("Final c:", c)
    print("Elapsed:", elapsed) 
    print("Difference between distributed and sequential:", np.sum(c-A@b))


