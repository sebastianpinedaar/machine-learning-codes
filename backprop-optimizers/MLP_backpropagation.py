import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def mse (y, y_hat):
    o = np.linalg.norm(y-y_hat)**2
    return o

X = np.matrix([[1, -1, -1],
               [1, -1,1],
               [1, 1,-1],
               [1, 1,1]])

Y= np.matrix([[0],[1],[1],[0]])

b1 = np.matrix(np.random.rand(3,2))
b2 = np.matrix(np.random.rand(3,1))

mse_list = []

for i in range(300):

        for j in range(X.shape[0]):
                
                #choosing stochastic sample
                idx = np.random.choice(X.shape[0],1)
                x = X[idx,]
                y = Y[idx]

                #feedforward
                u1 = x*b1
                z1=sigmoid(u1)
                z1_=np.hstack((z1, np.ones((x.shape[0],1))))
                u2= z1_*b2
                z2 = sigmoid(u2)   

                #calculating mean square error

                #back propagation
                error = -2*(y-z2)
                g2 = np.multiply(np.multiply(z2,(1-z2)),error)
                g1 = np.multiply(g2*b2[:-1,:].T, np.multiply(z1, (1-z1)))

                #gradient of the loss
                delta1 = x.T*g1
                delta2 = z1_.T*g2

                #updating parameters
                b1 = b1 - delta1
                b2 = b2 - delta2

        #making predictions over all the training data to calculate MSE
        Z1=sigmoid(X*b1)
        Z1_=np.hstack((Z1, np.ones((X.shape[0],1))))
        y_hat = sigmoid(Z1_*b2)    
        mse_ = mse(Y, y_hat)
        mse_list.append(mse_)


plt.plot(mse_list)
plt.ylabel("MSE")
plt.xlabel("iteration")
plt.title("Convergence")
plt.show()

