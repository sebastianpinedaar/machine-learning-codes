# coding: utf-8
# Machine Learning Lab - Exercise Sheet 1 
# Author: Sebastian Pineda Arango  
# ID: 246098
# Universität Hildesheim - Data Analytics Master 
# 
# 
# ### Word Count Program
# 
# 

# The objective in this part is to count the words in a text. To do that we have divided the task in different tasks, which are explained as follows:
# 
# 1. Open the text file
# 2. Read the lines
# 3. Merge lines in a text
# 4. Preprocess text using regular expressions
# 5. Split text in words
# 6. Filter words: stop words and length. We use words with length greater than 1, otherwise they are considered meaningless.
# 7. Select unique words (done using "set" data type of type)
# 8. Count words
# 9. Order words by count
# 10. Select top 10 words
# 11. Plot top 10 words


#Note: some help for working with regular expressions were takne from:
#https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
#https://stackoverflow.com/questions/252626/need-a-simple-regex-to-find-a-number-in-a-single-word
#https://www.regextester.com/97589
#https://stackoverflow.com/questions/34117950/filter-strings-by-regex-in-a-list

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

#1. Opening the text file
f =  open("README.txt", "r")

#2. Reading the lines
lines = f.readlines()

###Without text preprocessing
text = ""

#3. Merge lines in a text: Stacking lines into a single text
for l in lines:
    text = text + " " +l

#4. Preprocess text using regular expressions
text = text.lower() ###All lower case
text2 = re.sub(r'\<.*?\>','',text) #filtering everything between <>
text3 = re.sub(r'[^\w\s]',' ',text2) #filtering punctuation
text4 = re.sub(r'[0-9]',' ',text3) #filtering numbers
text5 = re.sub(r'\n',' ', text4) #filtering end line character
text6 = re.sub(r'\_', ' ', text5) #filter underline

#5. Splitting text into words. The words are considered as the group of characters separated by space (" ")
words = text6.split(" ")

#list of stop words
stop_words = ["the", "an", "and", "be", "to", "https", "for", "of", "on", "com", "with", "this", "in"]

#6. Filtering only words which length greater than one
words1 = filter(lambda x: len(x)>1 , words)

#6. Filtering only words that are non-stop words
words2 = filter(lambda x: x not in stop_words, words1)

#7. Getting unique words
unique_words = set(words2)

#creating dictionary to count
dict_count= {}

#8. counting words
for w in unique_words:
    dict_count[w]=words.count(w)

#getting the list of keys
keys = list(dict_count.keys())

#getting the list of values
values = list(dict_count.values())

#creating dataframe with keys and values as columns
df_count = pd.DataFrame({
    'word': keys,
    'count': values
})

#9-10. ordering data and selecting top data
df_top = df_count.sort_values(by=['count'], ascending=False).iloc[0:10]


#1plotting results of top appearing words
get_ipython().magic('matplotlib inline')

fig, ax = plt.subplots(figsize=(16, 10))

#barplot for histogram
ax.bar(list(df_top['word']),list(df_top['count']))
ax.grid()



#### Matrix Multiplication

#function reference in https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
A = np.matrix(np.random.random( (100,20))) #creating random matrix of size 100x2
v = np.matrix(np.random.normal(2, 0.01, (20,1))) #creating random vector of size 20x1

c=np.zeros((100,1)) #initializing matrix with the multiplication
for i in range(A.shape[0]): #iterating over rows
    for j in range( A.shape[1]):#iterating over columns
        c[i] = c[i] + A[i,j]*v[j] #performing cumulative multiplication

c_ = A*v #performing multiplication to serve as comparison
print("Proof of the operation. This number should be zero: %.4f"%sum(c-c_))

#calculating mean
mean = np.mean(c)

#calculating standard deviation
sd = np.std(c)

#plotting histogram
plt.hist(c, bins=5)
plt.grid()
plt.title("Histogram")

#printing mean and standarg deviation
print("The mean is : %.4f"% mean)
print("The standard deviation is %.4f"% sd)
print("The mean of A is: %.4f"% np.mean(A))
print("The mean of v is: %.4f"% np.mean(v))


#Creating data sets
A1 = np.matrix(np.random.normal(2, 0.01, (100,2)))
A2 = np.matrix(np.random.normal(2, 0.1, (100,2)))
A3 = np.matrix(np.random.normal(2, 1.0, (100,2)))


##LEARN-LINREG-NORMEQ
def learn_linreg_normeq(A):
    
    '''This function takes a two columns matrix A and usse the first column as predictor and the second one as target 
    to fit a basi linear regression model. The output es then the parameter vector (beta) which better fits the regression.'''
    
    #Separating columns
    x = A[:,0]
    y = A[:,1]
    
    #Adding column of ones
    x = np.hstack((x, np.ones(np.shape(x))))
    
    #Converting to matrix data type, so that it is easy to operate
    x = np.matrix(x)
    y = np.matrix(y)
    
    #applying the mathematical solution
    beta = (np.linalg.inv(x.T*x))*x.T*y
    
    return beta



beta_A1 = learn_linreg_normeq(A1)
beta_A2 = learn_linreg_normeq(A2)
beta_A3 = learn_linreg_normeq(A3)



print("Beta A1:",beta_A1.round(4))
print("Beta A2:",beta_A2.round(4))
print("Beta A3:",beta_A3.round(4))

#PREDICT-SIMPLE-LINREG
def predict_simple_linreg(beta,x):
    
    '''This function recieves to parameters: beta and x, to calculate the predictions of a basic linear regression model.'''
    
    #Organizing data to be of size = NX1
    x = np.reshape(x, (-1,1))
    
    #Adding new column
    x = np.hstack((x, np.ones(np.shape(x))))
    
    #Casting data
    x = np.matrix(x)
    beta = np.matrix(beta)
    
    #Applying matrix multiplication
    y_pred = x*beta
    
    return y_pred
    

#Making predictions for all A values. This implies to take x from A

x_A1 = A1[:,0] #Taking predictor from matrix A1
y_pred_A1 = predict_simple_linreg(beta_A1, x_A1)

x_A2 = A2[:,0] #Taking predictor from matrix A2
y_pred_A2 = predict_simple_linreg(beta_A2, x_A2)

x_A3 = A3[:,0] #Taking predictor form matrix A3
y_pred_A3 = predict_simple_linreg(beta_A3, x_A3)


#Creating a vector of continuous values to plot the predicted line
x_est_A1 = np.arange(min(x_A1[:,0]), max(x_A1[:,0]), (max(x_A1[:,0])-min(x_A1[:,0]))/1000)

#Making predictions over the vector 
y_est_A1 = predict_simple_linreg(beta_A1, x_est_A1)

#Creating objects to plot for results of matrix A1
fig, ax = plt.subplots(figsize=(16, 10))

ax.plot(x_est_A1, y_est_A1) # plotting estimated line
ax.plot(x_A1, y_pred_A1,'.')# plotting estimated values (predicted values)
ax.plot(x_A1, A1[:,1], 'o')# plotting real values
ax.grid()

ax.legend(("Estimated line", "Predicted values", "Real values"))

#Creating a vector of continuous values to plot the predicted line
x_est_A2 = np.arange(min(x_A2[:,0]), max(x_A2[:,0]), (max(x_A2[:,0])-min(x_A2[:,0]))/1000)
y_est_A2= predict_simple_linreg(beta_A2, x_est_A2)

#Creating objects to plot for results of matrix A2
fig, ax = plt.subplots(figsize=(16, 10))

ax.plot(x_est_A2, y_est_A2)# plotting estimated line
ax.plot(x_A2, y_pred_A2,'.')# plotting estimated values (predicted values)
ax.plot(x_A2, A2[:,1], 'o')# plotting real values
ax.grid()

ax.legend(("Estimated line", "Predicted values", "Real values"))


#Creating a vector of continuous values to plot the predicted line
x_est_A3 = np.arange(min(x_A3[:,0]), max(x_A3[:,0]), (max(x_A3[:,0])-min(x_A3[:,0]))/1000)
y_est_A3= predict_simple_linreg(beta_A3, x_est_A3)

#Creating objects to plot for results of matrix A2
fig, ax = plt.subplots(figsize=(16, 10))

ax.plot(x_est_A3, y_est_A3)# plotting estimated line
ax.plot(x_A3, y_pred_A3,'.')# plotting estimated values (predicted values)
ax.plot(x_A3, A3[:,1], 'o')# plotting real values
ax.grid()

ax.legend(("Estimated line", "Predicted values", "Real values"))

#Setting beta0 to 0
beta_A1[1]= 0
beta_A2[1]= 0
beta_A3[1]= 0

#Creating data to plot the estimated line for A1 dataset
x_est_A1 = np.arange(min(x_A1[:,0]), max(x_A1[:,0]), (max(x_A1[:,0])-min(x_A1[:,0]))/1000)
y_est_A1 = predict_simple_linreg(beta_A1, x_est_A1)

fig, ax = plt.subplots()

#Plotting the line of A1 data set
ax.plot(x_est_A1, y_est_A1,'.')
ax.plot(x_A1, A1[:,1], 'o')
ax.grid()

#Creating data to plot the estimated line for A2 dataset
x_est_A2 = np.arange(min(x_A2[:,0]), max(x_A2[:,0]), (max(x_A2[:,0])-min(x_A2[:,0]))/1000)
y_est_A2= predict_simple_linreg(beta_A2, x_est_A2)

fig, ax = plt.subplots()

#Plottting the line of A2 data set
ax.plot(x_est_A2, y_est_A2,'.')
ax.plot(x_A2, A2[:,1], 'o')
ax.grid()

#Creating data to plot the estimated line for A3 dataset
x_est_A3 = np.arange(min(x_A3[:,0]), max(x_A3[:,0]), (max(x_A3[:,0])-min(x_A3[:,0]))/1000)
y_est_A3= predict_simple_linreg(beta_A3, x_est_A3)

fig, ax = plt.subplots()

#Plotting the line of A3 data set
ax.plot(x_est_A3, y_est_A3,'.')
ax.plot(x_A3, A3[:,1], 'o')
ax.grid()


#Recalculating beta values
beta_A1 = learn_linreg_normeq(A1)
beta_A2 = learn_linreg_normeq(A2)
beta_A3 = learn_linreg_normeq(A3)

#Setting beta1 to 0
beta_A1[0]= 0
beta_A2[0]= 0
beta_A3[0]= 0

#Creating data to plot the estimated line for A1 dataset
x_est_A1 = np.arange(min(x_A1[:,0]), max(x_A1[:,0]), (max(x_A1[:,0])-min(x_A1[:,0]))/1000)
y_est_A1 = predict_simple_linreg(beta_A1, x_est_A1)

fig, ax = plt.subplots()

#Plotting the line of A1 data set
ax.plot(x_est_A1, y_est_A1,'.')
ax.plot(x_A1, A1[:,1], 'o')
ax.grid()

#Creating data to plot the estimated line for A2 dataset
x_est_A2 = np.arange(min(x_A2[:,0]), max(x_A2[:,0]), (max(x_A2[:,0])-min(x_A2[:,0]))/1000)
y_est_A2= predict_simple_linreg(beta_A2, x_est_A2)

fig, ax = plt.subplots()

#Plottting the line of A2 data set
ax.plot(x_est_A2, y_est_A2,'.')
ax.plot(x_A2, A2[:,1], 'o')
ax.grid()

#Creating data to plot the estimated line for A3 dataset
x_est_A3 = np.arange(min(x_A3[:,0]), max(x_A3[:,0]), (max(x_A3[:,0])-min(x_A3[:,0]))/1000)
y_est_A3= predict_simple_linreg(beta_A3, x_est_A3)

fig, ax = plt.subplots()

#Plotting the line of A3 data set
ax.plot(x_est_A3, y_est_A3,'.')
ax.plot(x_A3, A3[:,1], 'o')
ax.grid()


def learn_using_library(A):
    
    '''This function takes a two columns matrix A and usse the first column as predictor and the second one as target 
    to fit a basic linear regression model using a numpy function.
    The output es then the parameter vector (beta) which better fits the regression.'''
    
    #Taking out the x and y vector
    x = A[:,0]
    y = A[:,1]
    
    #Creating the x matrix
    x = np.hstack((x, np.ones(np.shape(x))))
    
    #Casting the matrix
    x = np.matrix(x)
    y = np.matrix(y)
    
    #Using the numpy function to find the beta value
    beta = np.linalg.lstsq(x,y)
    
    return beta

beta_A1_numpy = learn_using_library(A1)
beta_A2_numpy = learn_using_library(A2)
beta_A3_numpy = learn_using_library(A3)

print("Beta A1 with numpy:", beta_A1_numpy[0].round(4))
print("Beta A2 with numpy:", beta_A2_numpy[0].round(4))
print("Beta A3 with numpy:", beta_A3_numpy[0].round(4))


beta_A1 = learn_linreg_normeq(A1)
beta_A2 = learn_linreg_normeq(A2)
beta_A3 = learn_linreg_normeq(A3)

print("Beta A1 with normal equations:", beta_A1_numpy[0].round(4))
print("Beta A2 with normal equations:", beta_A2_numpy[0].round(4))
print("Beta A3 with normal equations:", beta_A3_numpy[0].round(4))

