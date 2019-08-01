import numpy as np 
from mpi4py import MPI
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import sys
from sklearn import datasets
import matplotlib.pyplot as plt


##defining functions

def find_assignation(X,C):
    
    '''Finds the cluster represented by centers in C to which each 
       sample of X belongs. Returns the number of the cluster for 
       each sample.'''
    
    assignation=[]
    for i in range(X.shape[0]):
        #Look for the closes center for each data point
        assignation.append(np.argmin(np.linalg.norm(np.subtract(X[i,].reshape(1,-1),C),axis=1)))

    return assignation

def find_center(X, assignation, k):
    
    '''Returns the new centers according to the assignation. '''
    
    d = X.shape[1]
    C = np.zeros([k, d])
    
    for i in list(set(assignation)):
        
        #calculate the new center with assgination index given by i
        C[i,:] = X[np.array(assignation)==i,:].sum(axis=0)
    
    return C

def find_MSE(X,C,assignation):
    
    '''Returns the RMSE for a given clustering configuration: (data and centers)'''
    
    q = []
    
    for i, a in enumerate(assignation):
        q.append(np.linalg.norm(X[i,]-C[a,])**2)
    
    return np.sum(q)

#setting variables for MPI communication
comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()

#reading arguments
K = int(sys.argv[1])
tol=1e-9
debug = True

if worker == 0:

    init = MPI.Wtime()

    #Initializing importnate variables for control
    stop_worker = [False]*num_workers
    MSE_list = [0]*num_workers
    MSE_diff = [1]*num_workers
    MSE_history = []

    #readint the dataset
    newsgroups = fetch_20newsgroups(subset='all')

    #transforming the dataset
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(newsgroups.data)

    #reducing the dimensionality of the dataset
    svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
    data = svd.fit_transform(X)
    

    if debug:
        print("Original data size: ", X.shape)
        print("Data size transformed: ", data.shape)

    #extracting information from the dataset
    n_samples = data.shape[0] #number of samples
    d = data.shape[1] #dimensionality of the problem

    #random centers generation
    idx = np.random.randint(0, n_samples, K)
    C= data[idx, ]
 
    #fraction of data for workers
    frac = int(n_samples/(num_workers-1))
    chunk_size = []

    print("Time preprocessing: ", MPI.Wtime()-init)
    
    i=0

    init =  MPI.Wtime()
    #sending data to workers
    for i in range(1, num_workers-1):
        comm.send(data[((i-1)*frac):(i*frac)], dest=i)
        chunk_size.append(frac)
    comm.send(data[(i*frac):], dest=(i+1))
    chunk_size.append(data[(i*frac):].shape[0]) 

    print("Time sending initial data: ", MPI.Wtime()-init)

else:
    #receiving data
    data = comm.recv()

#initializing control variables for the iterative center calculation
stop = False
max_iter = 20
local_MSE=0
i = 0
times = []
while  stop==False and i<max_iter:

    if worker==0:
        
        init = MPI.Wtime()        
        i +=1

        #evaluating if all workers have converged
        if(sum(stop_worker)==(num_workers-1)):
            stop=True
            np.save("C",C)
       
        #sending center and "stop" flag to workers
        for w in range(1, num_workers):
            buf = {'centers':C,
                'stop': stop}
            comm.send(buf, dest=w)

        #receiving partial centers and aggregating to final resutls
        centers = np.zeros(C.shape)
        count = np.zeros(K)
        for w in range(1, num_workers):

            #checking if the worker has converged
            if(abs(MSE_diff[w])<tol):
                stop_worker[w]=True

            #receiving information from worker
            buf = comm.recv(source=w)

            #reading MSE variables
            MSE_local = buf["local_MSE"]
            MSE_diff[w] = MSE_list[w]-MSE_local
            MSE_list[w] = MSE_local

            #agreggating centers sum and counts
            centers += buf["local_centers"]
            count += np.array(buf["local_count"])
        
        #calculating current MSE
        current_MSE = sum(MSE_list)/n_samples
        MSE_history.append(current_MSE)
        print("MSE at iter ",i, ": ", current_MSE)

        #computing centroids
        count[count==0] = 1 #to avoid zero division
        C = np.divide(centers.T, count).T #mean of centers: centroids
        
        final = MPI.Wtime()

        times.append(final-init)

    else:
        
        i +=1 #counter of the worker

        buff = comm.recv( source=0) #recigin information of current centers and stop flag
        C = buff["centers"]
        stop = buff["stop"]

        #computing local centers
        assignation = find_assignation(data,C) #finding the clusters of every data point
        local_centers = find_center(data, assignation, K) #finding local centers using local assignation
        local_MSE = find_MSE(data,C,assignation) #finding local MSE
        local_count = [assignation.count(a) for a in range(K)] #finging local count

        #creating dictionary to send to the master
        buf = {
            "local_centers": local_centers,
            "local_MSE": local_MSE,
            "local_count": local_count
        }

        #sending dictionary to the master
        comm.send(buf, dest=0)


if worker==0:
    print("Time per iteration:", times)
    print("Average time:", np.mean(times))
    print("Standard deviation:", np.std(times))
