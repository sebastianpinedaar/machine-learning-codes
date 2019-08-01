#Author: Sebastian Pineda Arango
#Institution: University of Hildesheim
#Year: 2019
#Functionality: This script should be called with mpiexec and is
#able to multiply tWo randomly create matrices using several process
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
N= int(sys.argv[1])

init = MPI.Wtime()

#initizing data in master and workers
if worker ==0:

    #Creating the random matrices to multiply
    A = np.random.randint(1,10,(N,N)).astype(int)
    B = np.random.randint(1,10,(N,N)).astype(int)
    C = np.zeros ((N,N)).astype(int)

else:
    A = None
    B = None

#broadcasting matrices
A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

#fraction of data to be received by workers
data_frac = int(N/num_workers)

#Computing the elements to be gahered by master. The element for the last worker is
#different because the number of data may be not even distributable
if worker != (num_workers-1):
    C_partial = {'C':A[worker*data_frac:(worker+1)*data_frac,]@B,
             'owner':worker}
else:
    C_partial = {'C':A[worker*data_frac:,]@B,
             'owner':worker}

#gathering data on master
data = comm.gather(C_partial, root=0)

#processing gathered data on master
if worker==0:

    #iterating over the gathered data
    for element in data:

        #interpreting the gathered dat
        owner = element['owner']
        C_partial = element['C']
     
        if owner != (num_workers-1):
            C[owner*data_frac:(owner+1)*data_frac,:]=C_partial
        else:
            C[owner*data_frac:,:]=C_partial      

    #calculating elapsed time
    elapsed = MPI.Wtime() - init
    
    print("Final C: ", C)
    print("Time elapsed: ",elapsed)

    #check of results
    print("Difference in result between distributed and sequential:", np.sum(C-A@B))

