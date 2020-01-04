from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Activation
from sklearn.utils import shuffle
import pickle

class Varkeys(Layer):

    def __init__(self, embedding_dim, n_keys, values, num_classes, **kwargs):

        self.output_dim = embedding_dim
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        #self.initializer = keras.initializers.random_uniform([dict_size, keysize],maxval=1)
        self.values =  tf.constant(values, dtype=tf.float32, shape = (n_keys, num_classes))
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim 
        self.dict_size = n_keys
        super(Varkeys, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.keys = self.add_weight(name='keys', 
                                      shape=(self.dict_size, self.embedding_dim),
                                      initializer=self.initializer,
                                      trainable=True)
        
        super(Varkeys, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)
        KV_ = tf.diag(tf.reshape(tf.reciprocal( tf.matmul(KV,tf.ones((self.num_classes,1)))) , [-1]))
        output = tf.matmul(KV_, KV)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

    
    def sq_distance(self, A, B):
        print('im in distance function')
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


    def kernel (self, A,B):
        print('im in kernel function!!')
        d = self.sq_distance(A,B)
        o = tf.reciprocal(d+1)
        return o
    
    def kernel_cos(self, A,B):
      
        normalize_A = tf.nn.l2_normalize(A,1)        
        normalize_B = tf.nn.l2_normalize(B,1)
        cossim = tf.matmul(normalize_B, tf.transpose(normalize_A))
        return tf.transpose(cossim)
      
    def kernel_gauss(self, A,B):
        d = self.sq_distance(A,B)
        o = tf.exp(-(d)/100)
        return o

    def set_values (self, values):
        self.values =  tf.constant(values, dtype=tf.float32, shape = (self.n_keys, self.num_classes))


class RelaxedSimilarity(Layer):

    def __init__(self, emb_size, n_centers, n_classes, gamma=0.1, **kwargs):
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        self.emb_size = emb_size
        self.gamma = gamma
        self.n_centers = n_centers
        self.n_classes = n_classes
        super(RelaxedSimilarity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.keys = self.add_weight(name='keys', 
                                      shape=(self.n_centers, self.emb_size, self.n_classes),
                                      initializer=self.initializer,
                                      trainable=True)     
        super(RelaxedSimilarity, self).build(input_shape)

    def call(self, X):

        X_n = tf.math.l2_normalize(X, axis=1)
        W_n = tf.math.l2_normalize(self.keys, axis=1)
        inner_logits = tf.einsum('ie,kec->ikc', X_n, W_n)
        inner_SoftMax = tf.nn.softmax((1/self.gamma)*inner_logits, axis=1)
        output = tf.reduce_sum( tf.multiply(inner_SoftMax, inner_logits), axis=1)
        return output

def custom_loss(layer, sigma=0.01, custom=1):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):

      if(custom==1):
        return keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)+sigma*tf.reduce_sum(layer.kernel(layer.keys, layer.keys))# + sigma*tf.reduce_mean(layer.kernel(layer.keys, layer.keys) , axis=-1)
      else:
        return keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
   
    # Return a function
    return loss
    

def SoftTripleLoss(layer, lamb=5, delta=0.01):

    def loss(y_true, y_pred):
      s = lamb*(y_pred - delta*y_true)
      outer_SoftMax = tf.nn.softmax(s)
      soft_triple_loss = -tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(outer_SoftMax, y_true), axis=1)))
      return soft_triple_loss
    return loss

def sample_train(x_train, y_train, pct):

    print("Train_pct=", pct)
    n_train = x_train.shape[0]
    idx = np.arange(n_train)
    np.random.shuffle(idx)

    train_samples = int(pct*n_train)
    x_train_pct = x_train[idx][:train_samples]
    y_train_pct = y_train[idx][:train_samples]

    return x_train_pct, y_train_pct

def construct_models (model, embedding_dim, n_keys, values, num_classes, lr, sigma):

    if model == "RESNET":

        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
        input = layers.Input(shape=( 32,32,3,))
        x=layers.UpSampling2D((2,2) )(input)
        x=layers.UpSampling2D((2,2))(x)
        x=layers.UpSampling2D((2,2))(x)
        x=conv_base(x)
        x=layers.Flatten()(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dense(512, activation='relu')(x)
        x=layers.Dropout(0.5)(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dense(embedding_dim, activation='relu')(x)
        x=layers.BatchNormalization()(x)

        varkeys_output = Varkeys(embedding_dim, n_keys, values, num_classes)(x)
        plain_output = layers.Activation('softmax')(layers.Dense(num_classes)(x))

        plain_model = Model(inputs=input, outputs=plain_output)
        varkeys_model = Model(inputs=input, outputs=varkeys_output)


        varkeys_model.compile(loss=custom_loss(varkeys_model.layers[-1], sigma, 1),#keras.losses.categorical_crossentropy,
                    # optimizer=keras.optimizers.SGD(lr=0.1),
                    optimizer = keras.optimizers.rmsprop(lr=lr, decay=1e-6),
                    metrics=['accuracy'])

        plain_model.compile(loss= keras.losses.categorical_crossentropy,#keras.losses.categorical_crossentropy,
                    # optimizer=keras.optimizers.SGD(lr=0.1),
                    optimizer = keras.optimizers.rmsprop(lr=lr, decay=1e-6),
                    metrics=['accuracy'])


    else:

        layers_dim=[32, 64, 512]
        input = layers.Input(shape=( 32,32,3,))
        x = layers.Conv2D(layers_dim[0], (3, 3), padding='same', input_shape=[32,32,3])(input)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(layers_dim[0], (3,3))(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(layers_dim[0], (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(layers_dim[1], (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(layers_dim[1], (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(layers_dim[2])(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(embedding_dim)(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        varkeys_output = Varkeys(embedding_dim, n_keys, values, num_classes)(x)
        plain_output = Activation('softmax')(layers.Dense(num_classes)(x))

        plain_model = Model(inputs=input, outputs=plain_output)
        varkeys_model = Model(inputs=input, outputs=varkeys_output)

    return varkeys_model, plain_model


def print_params(model, embedding_dim, n_keys, values, num_classes, lr, sigma, batch_size, epochs, dataset, input_shape, patience):

    print(  "embedding_dim   =  ", embedding_dim, "\n",
            "n_keys          =  ", n_keys, "\n",
            "num_classes     =  ", num_classes, "\n",
            "batch_size      =  ", batch_size, "\n",
            "lr              =  ", lr, "\n",
            "epochs          =  ", epochs, "\n",
            "sigma           =  ", sigma, "\n",
            "n_output        =  ", num_classes, "\n",
            "model           =  ", model, "\n",
            "dataset         =  ", dataset, "\n",
            "input_shape     =  ", input_shape, "\n",
            "patience        =  ", patience)
