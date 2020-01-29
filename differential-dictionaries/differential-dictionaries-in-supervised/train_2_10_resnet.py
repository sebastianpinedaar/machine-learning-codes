import models as m
from params import *
import automatic_dataset_loader as adl
import matplotlib.pyplot as plt
import pickle

epochs = 500
patience = 20
lr = 0.00001
sigma = 0
kernel_type = "gauss"
n_trials = 1
train_pct = 0.10
filepath = "model10.h5"
model = "RESNET"
num_classes = 10

history = []

x_train, x_test, y_train, y_test = adl.load_CIFAR10()
x_train_pct, y_train_pct = m.sample_train(x_train, y_train, train_pct)
m.print_params(model, embedding_dim, n_keys_per_class, num_classes, lr, sigma, batch_size, epochs, dataset, input_shape, patience)

for i in range(n_trials):

  varkeys_model, plain_model, embeddings = m.construct_models(model, embedding_dim, n_keys_per_class, num_classes, lr, sigma)

  #model = construct_model_STL("CNN", embedding_dim, n_keys_per_class, num_classes, lr, gamma)

  callbacks = [m.EarlyStopping(monitor='val_loss', patience=patience)]
  callbacks2 = [m.EarlyStopping(monitor='val_loss', patience=patience), m.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')]

 
  history_plain = plain_model.fit(x_train_pct, y_train_pct,
                          batch_size=  batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test),
                          callbacks = callbacks2)

  
  #without initialization
  
  """	
  plain_model.load_weights(filepath)
  history_vk = varkeys_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)
  """
  #with init k means
  plain_model.load_weights(filepath)
  init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_keys_per_class, num_classes, embedding_dim)
  varkeys_model.layers[-1].set_keys(init_keys)
  kmeans_acc = varkeys_model.evaluate(x_test, y_test)
  history_vk_1 = varkeys_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)

  #with init k medois
  """
  plain_model.load_weights(filepath)
  init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_keys_per_class, num_classes, embedding_dim, init_method= "KMEDOIDS")
  varkeys_model.layers[-1].set_keys(init_keys)
  kmedoids_acc = varkeys_model.evaluate(x_test, y_test)

  history_vk_2 = varkeys_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)
  """
  #gauss kernel k means
  varkeys_model, plain_model, embeddings = m.construct_models(model, embedding_dim, n_keys_per_class, num_classes, lr, sigma, kernel_type = "gauss")
  plain_model.load_weights(filepath)
  init_keys = m.get_initial_weights(embeddings, x_train_pct, y_train_pct, n_keys_per_class, num_classes, embedding_dim, init_method= "KMEANS")
  varkeys_model.layers[-1].set_keys(init_keys)
  history_gauss_kmeans = varkeys_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)

  #gauss kernel
  """
  varkeys_model, _, _ = m.construct_models(model, embedding_dim, n_keys_per_class, num_classes, lr, sigma, kernel_type = "gauss")
  history_gauss = varkeys_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)


  
  varkeys_model, _ , _ = m.construct_models(model, embedding_dim, n_keys_per_class, num_classes, lr, sigma)
  history_vk_no_init = varkeys_model.fit(x_train_pct, y_train_pct,
                  batch_size=  batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks)
  
  """

  highest_plain = np.max(history_plain.history["val_acc"])
  #highest_vk = np.max(history_vk.history["val_acc"])
  highest_vk_1 = np.max(history_vk_1.history["val_acc"])
  #highest_vk_2 = np.max(history_vk_2.history["val_acc"])
  #highest_vk_no_init = np.max(history_vk_no_init.history["val_acc"])
  highest_gauss_kmeans = np.max(history_gauss_kmeans.history["val_acc"])
  #highest_gauss = np.max(history_gauss.history["val_acc"])

  history.append({"plain": highest_plain,
                  #"vk": highest_vk,
                  "vk_kmeans": highest_vk_1,
                  #"vk_kmedoids": highest_vk_2,
                  #"k_means": kmeans_acc,
                  #"k_medoids": kmedoids_acc,
                  #"vk_no_init": highest_vk_no_init,
                  "gauss_means": highest_gauss_kmeans})
                  #"gauss": highest_gauss})
  

  with open("train_2_"+model+str(int(train_pct*100))+"_trial_"+str(i), "wb") as f:
    pickle.dump(history, f)
