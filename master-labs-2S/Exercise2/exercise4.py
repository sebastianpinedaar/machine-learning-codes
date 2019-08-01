#importing libraries
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
  

#defining functions
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
    tokens = word_tokenize(x)
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    x = [porter.stem(word) for word in words]

    return x



def count_words (text):

    '''This function counts the number of ocurrences for every word in the text'''

    #number of words
    n_words  = len(text)

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
    #and addition of the dictionaries with DF
    n_docs = 0
    idf_dict_cpt = dict()
    for i in range(1, num_workers):
        tfidf = comm.recv(source=i)

        n_docs_ = tfidf["n_docs"]
        idf_dict_ = tfidf["idf"]

        
        n_docs += n_docs_
        idf_dict_cpt = add_dict(idf_dict_cpt, idf_dict_)

    #computation of the final IDF
    for key in idf_dict_cpt.keys():
        idf_dict_cpt[key] = np.log(n_docs/idf_dict_cpt[key])


else:

    #preprocessing
    news = read_news(folders)
    porter = PorterStemmer()
    en_stop = get_stop_words('en')

    docs = news.Text.apply(lambda x: preprocess_text(x, en_stop, porter))

    #computing term frequency
    tf_list = []
    for text in docs:
        
        tf_worker= count_words(text)
        tf_list.append(tf_worker)

    #computing document frequency
    idf_dict = {}
    for tf in tf_list:
        for key in tf.keys():
            if key in idf_dict.keys():
                idf_dict[key]+=1
            else:
                idf_dict[key]=1

    #sending back resultss
    comm.send({"n_docs": len(tf_list),
                "idf": idf_dict}, dest=0 )

    idf_dict_cpt = None

#broadcasting the final idf dictionary
idf_dict_cpt = comm.bcast(idf_dict_cpt, root=0)


if worker==0:

    #combining the tf-idf received from workers
    tfidf_list = []
    for i in range(1, num_workers):
        tfidf = comm.recv(source=i)
        tfidf_list.append(tfidf)

    final = MPI.Wtime()
    print("Elapsed time : ", final-init)

    with open("tfidf", 'w') as fp:
        json.dump(tfidf_list,fp)

else:

    #computing the final tf-idf for the assigned documents
    for tf_doc in tf_list:
        for key in tf_doc.keys():
            tf_doc[key] = tf_doc[key]*idf_dict_cpt[key]

    comm.send({"tfidf": tf_list}, dest=0 )

