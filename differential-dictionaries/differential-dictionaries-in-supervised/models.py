from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import KMeans


class Varkeys_Deprecated(Layer):

    def __init__(self, embedding_dim, n_keys, values, num_classes, kernel_type,  **kwargs):

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

class Varkeys(Layer):

    def __init__(self, embedding_dim, n_keys_per_class, num_classes, kernel_type = "inverse", **kwargs):

        self.output_dim = embedding_dim
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        #self.initializer = keras.initializers.random_uniform([dict_size, keysize],maxval=1)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim 
        self.n_keys = n_keys_per_class*num_classes
        self.n_keys_per_class = n_keys_per_class
        self.kernel_type = kernel_type
        super(Varkeys, self).__init__(**kwargs)

    def build(self, input_shape):

        self.keys = self.add_weight(name='keys', 
                                    #shape=(self.n_centers, self.emb_size, self.n_classes),
                                      shape=(self.n_keys_per_class, self.num_classes,  self.embedding_dim),
                                      initializer=self.initializer,
                                      trainable=True)     

        super(Varkeys, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        keys2 = tf.reshape(self.keys, (-1, self.embedding_dim))

        if (self.kernel_type == "gauss"):
          K= self.kernel_gauss(keys2, x)
        else:
          K= self.kernel(keys2, x)
        inner_logits = tf.transpose(tf.reduce_sum(tf.reshape(K, (self.n_keys_per_class, self.num_classes, -1)), axis=0))
        sum_inner_logits = tf.reduce_sum(inner_logits, axis=1)
        output = inner_logits / tf.reshape(sum_inner_logits, (-1, 1))

        return output

    
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
        o = tf.exp(-d/100)
        return o

    def get_keys(self):

        return self.get_weights()[0]
    
    def set_keys(self, keys):

        self.set_weights([keys])

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
        flatten_keys = keys2 = tf.reshape(layer.keys, (-1, layer.embedding_dim))
        return keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)+sigma*tf.reduce_sum(layer.kernel(flatten_keys, flatten_keys))# + sigma*tf.reduce_mean(layer.kernel(layer.keys, layer.keys) , axis=-1)
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

def construct_models (model, embedding_dim, n_keys_per_class, num_classes, lr, sigma, kernel_type = "inverse"):

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

        varkeys_output = Varkeys(embedding_dim, n_keys_per_class, num_classes, kernel_type)(x)
        plain_output = layers.Activation('softmax')(layers.Dense(num_classes)(x))

        plain_model = Model(inputs=input, outputs=plain_output)
        varkeys_model = Model(inputs=input, outputs=varkeys_output)
        embeddings = Model(inputs=input, outputs=x)

        varkeys_model.compile(loss=custom_loss(varkeys_model.layers[-1], sigma, 1),#keras.losses.categorical_crossentropy,
                    # optimizer=keras.optimizers.SGD(lr=0.1),
                    optimizer = optimizers.RMSprop(lr=lr),
                    metrics=['accuracy'])

        plain_model.compile(loss= keras.losses.categorical_crossentropy,#keras.losses.categorical_crossentropy,
                    # optimizer=keras.optimizers.SGD(lr=0.1),
                    optimizer = optimizers.RMSprop(lr=lr),
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

        varkeys_output = Varkeys(embedding_dim, n_keys_per_class, num_classes, kernel_type)(x)
        plain_output = layers.Activation('softmax')(layers.Dense(num_classes)(x))

        plain_model = Model(inputs=input, outputs=plain_output)
        varkeys_model = Model(inputs=input, outputs=varkeys_output)
        embeddings = Model(inputs=input, outputs=x)

        varkeys_model.compile(loss=custom_loss(varkeys_model.layers[-1], sigma, 1),#keras.losses.categorical_crossentropy,
            # optimizer=keras.optimizers.SGD(lr=0.1),
            optimizer = optimizers.RMSprop(lr=lr),
            metrics=['accuracy'])

        plain_model.compile(loss= keras.losses.categorical_crossentropy,#keras.losses.categorical_crossentropy,
                    # optimizer=keras.optimizers.SGD(lr=0.1),
                    optimizer = optimizers.RMSprop(lr=lr),
                    metrics=['accuracy'])

    return varkeys_model, plain_model, embeddings


def construct_model_STL(model, embedding_dim, n_keys_per_class, num_classes, lr, gamma):


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

        x = RelaxedSimilarity(embedding_dim, n_keys_per_class, num_classes, gamma)(x)
        output = layers.Softmax()(x)
        model = Model(inputs=input, outputs=output)

        model.compile(optimizer=optimizers.RMSprop(lr=lr), loss=SoftTripleLoss(model.layers[-2]), metrics=['acc'])

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

        x = RelaxedSimilarity(embedding_dim, n_keys_per_class, num_classes, gamma)(x)
        output = layers.Softmax()(x)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizers.RMSprop(lr=lr), loss=SoftTripleLoss(model.layers[-2]), metrics=['acc'])

  return model



def print_params(model, embedding_dim, n_keys_per_class, num_classes, lr, sigma, batch_size, epochs, dataset, input_shape, patience):

    print(  "embedding_dim   =  ", embedding_dim, "\n",
            "keys_per_class= =  ", n_keys_per_class, "\n",
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
    
def get_initial_weights(embedding, x, y, n_clusters, num_classes, embedding_dim, init_method= "KMEANS", n_init=20, max_iter=200):


    embedding_outputs = embedding.predict(x)
    centers = np.zeros([n_clusters, num_classes, embedding_dim])
    
    if init_method=="KMEANS":
        

        for i in range(num_classes):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init= n_init, max_iter =max_iter)
            x_class = embedding_outputs[np.where(np.argmax(y,1)==i)]
            centers[:, i, :] = kmeans.fit(x_class).cluster_centers_

    elif init_method =="KMEDOIDS":

        for i in range(num_classes):
            x_class = embedding_outputs[np.where(np.argmax(y,1)==i)]
            initial_medoids = np.random.randint(low=0, high= x_class.shape[0], size=n_clusters)
            cluster = kmedoids(x_class, initial_medoids)
            centers_idx = cluster.process().get_medoids()
            centers[:, i, :] = embedding_outputs[centers_idx]


    else:
        raise "Not implemented"

    return centers