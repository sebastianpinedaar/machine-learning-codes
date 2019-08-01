import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def mse (y, y_hat):
    o = np.linalg.norm(y-y_hat)**2
    return o

def predict(X, b1, b2):
    Z1=sigmoid(X*b1)
    Z1_=np.hstack((Z1, np.ones((X.shape[0],1))))
    y_hat = sigmoid(Z1_*b2)    
    return y_hat

def grad(x, y, b1, b2):

    #feedforward
    u1 = x*b1
    z1=sigmoid(u1)
    z1_=np.hstack((z1, np.ones((x.shape[0],1))))
    u2= z1_*b2
    z2 = sigmoid(u2)   


    #back propagation
    error = -2*(y-z2)
    g2 = np.multiply(np.multiply(z2,(1-z2)),error)
    g1 = np.multiply(g2*b2[:-1,:].T, np.multiply(z1, (1-z1)))

    #gradient of the loss
    grad1 = x.T*g1
    grad2 = z1_.T*g2

    return grad1, grad2


def nester(x, y, b1, b2, alpha, v1, v2, epsilon):
    
    """Implementation of Nesterov Momentum"""
    
    b1_tilde = b1 + alpha*v1
    b2_tilde = b2 + alpha*v2
    
    grad1, grad2 = grad(x,y, b1_tilde, b2_tilde)
    
    v1 = alpha*v1 - epsilon*grad1
    v2 = alpha*v2 - epsilon*grad2
    
    b1 = b1 + v1
    b2 = b2 + v2
    
    return b1, b2, v1, v2

def adam(x, y, b1, b2, phi1, phi2, epsilon, delta, t, s, r):
    
    """Implementation of adam optimizer """
    
    grad_dict = {}
    grad_dict["1"], grad_dict["2"] = grad(x,y, b1, b2)
    
    s_hat = {}
    r_hat = {}
    
    for i in list(grad_dict.keys()):
        s[i] = phi1*s[i] + (1-phi1)*grad_dict[i]
        s_hat[i] = s[i]/(1-phi1**t)
        
        r[i]= phi2*r[i] + (1-phi2)*np.multiply(grad_dict[i], grad_dict[i])
        r_hat[i] = r[i]/(1-phi2**t)
        
    b1 = b1-epsilon*(np.multiply(s_hat["1"], 1/(delta+np.sqrt(r_hat["1"]))))
    b2 = b2-epsilon*(np.multiply(s_hat["2"], 1/(delta+np.sqrt(r_hat["2"]))))
    

    return b1, b2, s, r


X = np.matrix([[1, -1, -1],
               [1, -1,1],
               [1, 1,-1],
               [1, 1,1]])

Y= np.matrix([[0],[1],[1],[0]])

b1 = np.matrix(np.random.rand(3,2))
b2 = np.matrix(np.random.rand(3,1))

b1_nester = b1
b2_nester = b2

v1 = np.matrix(np.zeros((3,1)))
v2 = np.matrix(np.zeros((3,1)))

b1_adam = b1
b2_adam = b2

s = {"1": np.zeros(b1.shape),
     "2": np.zeros(b2.shape)}

r = {"1": np.zeros(b1.shape),
     "2": np.zeros(b2.shape)}


#parameters
mse_list = []
epsilon = 0.001
alpha = 0.5
delta = 1e-8
t = 0
phi1 = 0.9
phi2 = 0.999

for i in range(1000):

        for j in range(X.shape[0]):
            
                t += 1
                
                #choosing stochastic sample
                idx = np.random.choice(X.shape[0],1)
                x = X[idx,]
                y = Y[idx]

                #gradient
                grad1, grad2 = grad(x,y, b1, b2)
                
                #updating parameters by SGD
                b1 = b1 - epsilon*grad1
                b2 = b2 - epsilon*grad2

                #updating parameters using nesterov momentum
                b1_nester, b2_nester, v1,v2 = nester(x, y, b1_nester, b2_nester, alpha, v1, v2, epsilon)
                
                #updating parameters using adam optimizer
                b1_adam, b2_adam, s, r = adam(x, y, b1_adam, b2_adam, phi1, phi2, epsilon, delta, t, s, r)
                
        #making predictions over all the training data to calculate MSE
        y_hat = predict(X, b1, b2)
        y_hat_nester = predict(X, b1_nester, b2_nester)
        y_hat_adam = predict(X, b1_adam, b2_adam)
        
        #measuring the error
        mse_SGD = mse(Y, y_hat)
        mse_nester = mse(Y, y_hat_nester)
        mse_adam = mse(Y, y_hat_adam)
        
        mse_list.append([mse_SGD, mse_nester, mse_adam])

mse_list = np.array(mse_list)
plt.plot(mse_list[:,0])
plt.plot(mse_list[:,1])
plt.plot(mse_list[:,2])

plt.ylabel("MSE")
plt.xlabel("iteration")
plt.title("Convergence")
plt.legend(("SGD", "Nesterov", "Adam"))
plt.title("Convergence with learning rate="+str(epsilon))
plt.grid()
plt.show()

