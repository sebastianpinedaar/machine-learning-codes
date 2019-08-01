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


  
def add_dict(dict1, dict2):

    '''This function adds two dictionaries so that the common values
    of the common keys are added and the disjunct keys (keys that only
    one of the dictionaries has) are kept.'''

    for key1 in dict1: 
        if key1 in dict2:
            dict2[key1]+=dict1[key1]
        else:
            dict2[key1]=dict1[key1]
    return dict2
    

def read_news(folders, path):
    
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

    tokens = word_tokenize(x)
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words] 
    x = [porter.stem(word) for word in words]

    return x



def count_words (text):

    #number of words
    n_words = len(text)

    #unique workds
    unique_words = set(text)
    
    #creating dictionary to count
    dict_count= {}
    
    #8. counting words
    for w in unique_words:
        dict_count[w]=text.count(w)/n_words
    
    
    #getting the list of keys
    keys = list(dict_count.keys())
    
    #getting the list of values
    values = list(dict_count.values())
    
    #creating dataframe with keys and values as columns
    df_count = dict(zip(keys, values))

    
    return df_count

path = os.path.join("20_newsgroups","")

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()

init = MPI.Wtime()

if worker == 0:
    
    #the master sends the folders name to every worker

    folders = os.listdir(path)
    frac = int(len(folders)/(num_workers-1))

    i = 0
    for i in range(1, num_workers-1):
        comm.send(folders[((i-1)*frac):(i*frac)], dest=i)
    
    comm.send(folders[(i*frac):], dest=(i+1))
else:
    folders = comm.recv()


if worker == 0:
    
    #concatenation of the term frequency dictionaries sent by the workers
    #addition of the dictionaries with DF

    tf_list = []
    idf_dict = {}
    for i in range(1, num_workers):
        tfidf = comm.recv(source=i)

        tf_list_ = tfidf["tf"]
        idf_dict_ = tfidf["idf"]
        
        tf_list.append(tf_list_)
        idf_dict = add_dict(idf_dict, idf_dict_)

    n_docs = len(tf_list)

    #calculation of the final IDF for all the terms
    for key in idf_dict.keys():
        idf_dict[key] = np.log(n_docs/idf_dict[key])

    final = MPI.Wtime()
    print("Elapsed time : ", final-init)

    #saving results
    with open("idf_dictionary", 'w') as fp:
        json.dump(idf_dict,fp)

else:
    #reading files
    news = read_news(folders)

    #preprocessing
    porter = PorterStemmer()
    en_stop = get_stop_words('en')

    docs = news.Text.apply(lambda x: preprocess_text(x, en_stop, porter))

    #computing term frequency
    tf_list = []
    for text in docs:
        
        tf_worker= count_words(text)
        tf_list.append(tf_worker)

    #computing DF (for IDF)
    idf_dict = {}
    for tf in tf_list:
        for key in tf.keys():
            if key in idf_dict.keys():
                idf_dict[key]+=1
            else:
                idf_dict[key]=1

    #sending information to master
    comm.send({"tf": tf_list,
                "idf": idf_dict}, dest=0 )