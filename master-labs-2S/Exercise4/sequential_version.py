import numpy as np
import os
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from mpi4py import MPI
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import sys

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
    out = np.sqrt(np.mean(np.array(y_pred-y)**2))
    
    return out

def load_data(dataset):

    if dataset=="KDD_CUP":

        data = pd.read_csv("cup98LRN.txt", sep=",")
 
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        newdf = data.select_dtypes(include=numerics)
        features = list(newdf.columns)

        col_na = newdf.isna().mean()
        col_remove = list(col_na[col_na>0.3].index)
        for feature in col_remove:
            features.remove(feature)

        data_cleaned = data[features].dropna()
        features.remove("TARGET_D")

        X = np.array(data_cleaned[features])
        y = np.array(data_cleaned["TARGET_D"]).reshape(-1,1) 


    else:
        path = os.path.join("dataset","")
        files = os.listdir(path)
        data = []

        X, y = load_svmlight_file(path+"/"+files[0])
        X= X.todense()

        for file_name in files:

            X_, y_ = load_svmlight_file(path+"/"+file_name, n_features=479)
            X_= X_.todense()
            X = np.vstack((X, X_))
            y = np.vstack((y.reshape(-1,1), y_.reshape(-1,1)))
    
    return X, y

#initializing important variables
time_per_epoch = []
RMSE_train_history = []
RMSE_test_history = []
lr = 0.0000025
epochs = 100

name = sys.argv[1]

#reading datas
if(name[0]=="K"):
    X, y = load_data("KDD_CUP")
    print("Loading KDD CUP dataset...")
else:
    X, y = load_data("VIRUS")
    print("Loading virus dataset...")

X = np.hstack((np.ones((X.shape[0],1)),X))

#splitting data in train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

#initializing parameters
beta = np.matrix(np.random.rand(X.shape[1],1))

#scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#formatting data
X_train = np.matrix(X_train)
y_train = np.matrix(y_train)
X_test = np.matrix(X_test)
y_test = np.matrix(y_test)

n_train_samples = X_train.shape[0]
n_test_samples = X_test.shape[0]

#performing stochastic grdient descent
for epoch in range(epochs):

    init = time.time()
    for i in range(X_train.shape[0]):
        
        beta = beta - lr*grad(X_train[i,], y_train[i], beta)

    #saving RMSE information
    elapsed = time.time()-init
    time_per_epoch.append(elapsed)
    RMSE_train_history.append(linear_loss (X_train, beta, y_train))
    RMSE_test_history.append(linear_loss(X_test, beta, y_test ))

#formating RMSE information
RMSE_train_history = np.array(RMSE_train_history)
RMSE_test_history = np.array(RMSE_test_history)
time_per_epoch = np.array(time_per_epoch)

#wirintg information to disk for further analysis
np.save(name+"_RMSE_train", RMSE_train_history)
np.save(name+"_RMSE_test", RMSE_test_history)
np.save(name+"_time", time_per_epoch)
