import numpy as np
import os
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd


def grad(X, y, beta):

    '''This function implements the gradient of the logistic loss'''
    
    grad=-X.T*(y-X*beta)
    return grad

def linear_loss (X, beta, y):
    
    '''Computes the loss of linear regression (MSE).
   The parameters are:
    - X is the matrix of features
    - beta is the vector of parameters for the linear regression
    - y is the target vector     '''
    
    y_pred = X*beta
    out = np.sum(np.array(y_pred-y)**2)
    
    return out


def load_data(dataset):

    """This functions loads the dataset to use in the processing depending on the arguments"""

    if dataset=="KDD_CUP":

        #read data
        data = pd.read_csv("cup98LRN.txt", sep=",")
 
        #selecting on the non-numerical columns
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        newdf = data.select_dtypes(include=numerics)
        features = list(newdf.columns)

        #eliminating columns with many missing values
        col_na = newdf.isna().mean()
        col_remove = list(col_na[col_na>0.3].index)

        for feature in col_remove:
            features.remove(feature)

        #dropping rows with at least one missing value
        data_cleaned = data[features].dropna()

        #removing targets from the features columns
        features.remove("TARGET_D")
        features.remove("TARGET_B")

        #formatting output to array
        X = np.array(data_cleaned[features])
        y = np.array(data_cleaned["TARGET_D"]).reshape(-1,1) 


    else:

        #referecing directory with the data
        path = os.path.join("dataset","")
        files = os.listdir(path)
        data = []

        #reading files which are in svmlight format
        X, y = load_svmlight_file(path+"/"+files[0])
        X= X.todense()
        for file_name in files:

            X_, y_ = load_svmlight_file(path+"/"+file_name, n_features=479)
            X_= X_.todense()
            X = np.vstack((X, X_))
            y = np.vstack((y.reshape(-1,1), y_.reshape(-1,1)))
    
    return X, y


#initializing control variables for the iterative center calculation
stop = False
epochs = 100 #max number of epochs
local_MSE=0 #initial MSE
i = 0 #counter for iterations
times = [] #list for perfomance compasions
tol = 0.00001
test_size = 0.2

#initializing MPI communication
comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()

#lr = 0.00000000001
#lr = 0.00001
lr = 0.000001
name = sys.argv[1]


if worker==0:

    #reading data and splitting data among workers
    init = MPI.Wtime()

    if(name[0]=="K"):
        X, y = load_data("KDD_CUP")
        print("Loading KDD CUP dataset...")
    else:
        X, y = load_data("VIRUS")
        print("Loading virus dataset...")

    X = np.hstack((np.ones((X.shape[0],1)),X))

    #initializing flags and list to fill with perforamcne measurements
    stop_worker = [False]*num_workers
    MSE_train_list = [0]*num_workers
    MSE_test_list = [0]*num_workers
    MSE_train_diff = [1]*num_workers
    RMSE_train_history = []
    RMSE_test_history = []
    time_per_epoch = []

    #normalizing the data
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    frac_train = int(X_train.shape[0]/(num_workers-1))
    frac_test = int(X_test.shape[0]/(num_workers-1))

    #splitting data among workers
    i = 0 
    for i in range(1, num_workers-1):
        data = {
            'X_train': X_train[((i-1)*frac_train):(i*frac_train),:],
            'X_test':  X_test[((i-1)*frac_test):(i*frac_test),:],
            'y_train': y_train[((i-1)*frac_train):(i*frac_train)],
            'y_test': y_test[((i-1)*frac_test):(i*frac_test)]
        }
        comm.send(data, dest=i)
    
    data = {
        'X_train': X_train[(i*frac_train):,:],
        'y_train': y_train[(i*frac_train):],
        'X_test': X_test[(i*frac_test):,:],
        'y_test': y_test[(i*frac_test):]
    }    
    comm.send(data, dest=(i+1))

    n_samples = X.shape[0]
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]

    #initializing beta
    beta = np.matrix(np.random.rand(X.shape[1],1))

    final = MPI.Wtime()

    print("Reading and preprocessing time in master:", final-init)


else:

    #receiving data and formatting data 
    data = comm.recv(source=0)

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    X_train = np.matrix(X_train)
    X_test = np.matrix(X_test)
    y_train = np.matrix(y_train)
    y_test = np.matrix(y_test)

    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]


#parallel stochastic gradient descent
while stop==False:

    if worker==0:

        init = MPI.Wtime()
        i+=1
      
        #checkig whether the workers have stopped
        if (sum(stop_worker)==(num_workers-1)):
                stop=True

        #sending parameters to the workers
        for w in range(1, num_workers):
            data = {'beta':beta,
                        'stop': stop}
            comm.send(data, dest=w)

        #receving and agreggating data from workers
        beta_ = np.matrix(np.zeros((X.shape[1],1)))


        for w in range(1, num_workers):
            
            #checking if the change in the MSE is small
            if(abs(MSE_train_diff[w]/n_train_samples)<tol):
                stop_worker[w]=True

            #receiving data from worker
            buff = comm.recv(source=w)

            #setting flag to stop worker if the limit of epochs is met
            if(buff["iter"]>epochs):
                stop_worker[w]=True

            #saving local RMSE
            local_MSE_train = buff["local_MSE_train"]
            MSE_train_diff[w] = MSE_train_list[w]-local_MSE_train
            MSE_train_list[w] = local_MSE_train
            MSE_test_list[w] = buff["local_MSE_test"]

            #aggregating beta
            beta_ += buff["local_beta"]

        #computing current RMSE
        current_RMSE_train = np.sqrt(sum(MSE_train_list)/n_train_samples)
        current_RMSE_test = np.sqrt(sum(MSE_test_list)/n_test_samples)
        
        #appending current RMSE
        RMSE_train_history.append(current_RMSE_train)
        RMSE_test_history.append(current_RMSE_test)

        #averaging beta
        beta = beta_/(num_workers-1)

        final = MPI.Wtime()
        time_per_epoch.append(final-init)

    else:
        i+=1
    
        buff = comm.recv( source=0) #recigin information of current centers and stop flag
        beta = buff["beta"]
        stop = buff["stop"]

        #performng stochastic gradient descent locally
        for j in range(n_train_samples):
            
            beta = beta - lr*grad(X_train[j,:], y_train[j,], beta)

        #computing train and test
        local_MSE_train = linear_loss (X_train, beta, y_train)
        local_MSE_test = linear_loss (X_test, beta, y_test)

        #creating dictionary to send to the master
        buff = {
            "local_beta": beta,
            "local_MSE_train": local_MSE_train,
            "local_MSE_test": local_MSE_test,
            "iter": i
        }

        comm.send(buff, dest=0)

if worker==0:

    #saving information for further analysis
    RMSE_train_history = np.array(RMSE_train_history)
    RMSE_test_history = np.array(RMSE_test_history)
    time_per_epoch = np.array(time_per_epoch)

    np.save(name+"_RMSE_train", RMSE_train_history)
    np.save(name+"_RMSE_test", RMSE_test_history)
    np.save(name+"_time", time_per_epoch)