import models as m
from params import *
import automatic_dataset_loader as adl

x_train, x_test, y_train, y_test = adl.load_CIFAR10()
x_train_pct, y_train_pct = m.sample_train(x_train, y_train, 0.1)
m.print_params(model, embedding_dim, n_keys_per_class, num_classes, lr, sigma, batch_size, epochs, dataset, input_shape, patience)


varkeys_model, plain_model = m.construct_models(model, embedding_dim, n_keys_per_class, num_classes, lr, sigma)
#model = construct_model_STL("CNN", embedding_dim, n_keys_per_class, num_classes, lr, gamma)

callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]

varkeys_model.fit(x_train_pct, y_train_pct,
        batch_size=  batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks = callbacks)

scores = varkeys_model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (varkeys_model.metrics_names[1], scores[1]*100))