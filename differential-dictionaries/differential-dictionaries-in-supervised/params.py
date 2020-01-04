import numpy as np

#values
n_keys_per_class = 1

values = np.vstack((np.repeat([[1,0,0,0,0,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,1,0,0,0,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,1,0,0,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,1,0,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,1,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,1,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,0,1,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,1,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,0,1,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,0,0,1]], n_keys_per_class, axis=0)))


embedding_dim   = 64
n_keys          = values.shape[0]
num_classes     = values.shape[1]
batch_size      = 128
lr              = 0.0001
epochs          = 100
sigma           = 0.01
n_output        = num_classes
model           = "RESNET"
dataset         = "CIFAR-10"
input_shape     = [32,32,3]
patience        = 10