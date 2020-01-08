import numpy as np

import numpy as np

n_keys_per_class = 1
embedding_dim   = 64
num_classes     = 10
batch_size      = 32
lr              = 0.00001
epochs          = 3
sigma           = 0.01#only for Varkeys
n_output        = num_classes
model           = "CNN"
dataset         = "CIFAR-10"
input_shape     = [32,32,3]
patience        = 10
gamma           = 0.1 #only for STL
