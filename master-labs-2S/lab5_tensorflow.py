import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_olivetti_faces

# Load the faces datasets
data = fetch_olivetti_faces()
targets = data.target
data = data.images.reshape((len(data.images), -1))

#getting the numer of pixels
n_pixels = data.shape[1]

print("Data shape:", data.shape)
print("Targets shape:", targets.shape)
print("Num. of pixels:", n_pixels)

#splitting in train and test based on the code of [4]
train, test, targets_train, targets_test = train_test_split(
    data, targets, test_size=0.1, random_state=42)

y_train = train[:, n_pixels // 2:]
X_train = train[:, :(n_pixels + 1) // 2]
y_test = test[:, :(n_pixels + 1) // 2]
X_test = test[:, n_pixels // 2:]

n_features = X_train.shape[1]
n_output = y_train.shape[1]

def logistic_regression_model (lr, n_fetures, n_output, optimizer=tf.train.GradientDescentOptimizer):
    
    """Implementation of the multi-output logistic regression model"""
    
    X = tf.placeholder(tf.float32, shape = (None,n_features))
    Y = tf.placeholder(tf.float32, shape = (None,n_output))

    W = tf.Variable(tf.truncated_normal([n_features, n_output], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    output = tf.matmul(X,W)+b
    cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(    labels=Y, logits=output))

    train_step = optimizer(lr).minimize(cost)
    
    return X,Y, train_step, output, cost, W, b

def train_logistic( W, b, train_step, cost, n_iterations):
    
    """Training function for logistic model"""

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    cost_train_h = []
    cost_test_h = []
    
    for i in range(n_iterations):

        _, cost_train = sess.run([train_step, cost],
                           feed_dict = {X:X_train, Y:y_train})

        cost_test = sess.run(cost,
                           feed_dict = {X:X_test, Y:y_test})

        cost_train_h.append(cost_train)
        cost_test_h.append(cost_test)
            
    W_, b_ = sess.run([W, b], feed_dict = {X:X_test, Y:y_test})
    
    sess.close()

    return W_,b_, cost_train_h, cost_test_h

#Training the model
n_iterations = 200
lr = 1
X, Y, train_step, output, cost, W, b = logistic_regression_model (lr, n_features, n_output)
W, b, c_train, c_test = train_logistic(W, b, train_step, cost, n_iterations)

#Performing a test on a sample image
X_sample = X_train[1]
X_sample_reshaped = X_train[0].reshape((32, 64))
Y_pred = 1/(1+np.exp(-(X_sample@W+b)))
Y_pred = Y_pred.reshape((32, 64))
Y_true = y_train[1].reshape((32, 64))

plt.imshow(Y_pred)
plt.savefig('fig1.png')

plt.imshow(Y_true)
plt.savefig('fig2.png')

#initializing variables
n_iterations = 10
colors = ["b", "r", "g"]
lr_list = [0.01, 0.5, 1]
legend= []
optimizers = [tf.train.GradientDescentOptimizer, tf.train.AdamOptimizer, 
              tf.train.AdagradOptimizer]
optimizers_names =["GradientDescent", "Adam", "Adagrad"]

fig, ax = plt.subplots(len(optimizers), figsize=(10,10))

for i, opt in enumerate(optimizers):
    for lr,col in zip(lr_list, colors) :

        X, Y, train_step, output, cost, W, b = logistic_regression_model(lr, n_features, n_output, optimizer=opt)
        W, b, c_train, c_test = train_logistic(W, b, train_step, cost, n_iterations)
        ax[i].plot(c_test, col)
        ax[i].plot(c_train, col+"--")
        legend.append("Train lr="+str(lr))
        legend.append("Test lr="+str(lr))
    
    ax[i].legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
    ax[i].grid()
    ax[i].set_xlabel("Iterations")
    ax[i].set_ylabel("Loss")
    ax[i].set_title(optimizers_names[i])