import os
from stop_words import  get_stop_words
import re
import numpy as np 
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from mpi4py import MPI
from random import random
import time
import sys
import json


def read_news(folders):
    
    '''Function to read news located in given path folder'''
    
    class_list = []
    text_list = []

    for folder in folders:
        files = os.listdir(path+"/"+folder)

        for file_name in files:
            file = open(path+"/"+folder+"/"+file_name, 'r')
            text = file.read()
            class_list.append(folder)
            text_list.append(text)

    data = pd.DataFrame({'Text': text_list,
                        'Label': class_list})

    return data


def preprocess_text (x, stop_words, porter):
    
    '''This function preprocess a string, eliminateing sop words and
    non-alphanumeric characters.'''
    
    x = x.lower()
    tokens = word_tokenize(x) #tokenization
    words = [word for word in tokens if word.isalpha()]#cleaning
    words = [w for w in words if not w in stop_words]
    x = [porter.stem(word) for word in words]

    return x

path = os.path.join("20_newsgroups","")

#initializing variables for the communicaiton
comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()

init = MPI.Wtime()


if worker == 0:
    
    #the master sends the folders name to every worker

    folders = os.listdir(path) #path pointing to the directories
    frac = int(len(folders)/(num_workers-1))

    i = 0
    for i in range(1, num_workers-1):
        comm.send(folders[((i-1)*frac):(i*frac)], dest=i)
    
    comm.send(folders[(i*frac):], dest=(i+1))
else:

    #every worker receives the folder list 
    folders = comm.recv()


if worker == 0:
    
    #the master receives back the list of tokens
    #and append every list in a list of lists
    
    docs = []
    for i in range(1, num_workers):
        docs_ = comm.recv(source=i)
        docs.append(docs_)

    final = MPI.Wtime()
    print("Elapsed time : ", final-init)


else:

    #every worker reads the group of folders and perform preprocessing.
    #For the preprocessing, the no-alphabetic signs are removied and then
    #the text is tokenized.

    news = read_news(folders) #reading news
    porter = PorterStemmer() #object for stemming
    en_stop = get_stop_words('en') #list of stop words

    docs = news.Text.apply(lambda x: preprocess_text(x, en_stop, porter)) #going through all files in the directory
    comm.send(docs, dest=0 ) #sending back data to the master
