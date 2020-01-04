import models as m
from params import model, embedding_dim, n_keys, values, num_classes, lr, sigma, batch_size, epochs
import automatic_dataset_loader as adl

x_train, x_test, y_train, y_test = adl.load_CIFAR10()

x_train_pct, y_train_pct = m.sample_train(x_train, y_train, 0.1)

varkeys_model, plain_model = m.construct_models(model, embedding_dim, n_keys, values, num_classes, lr, sigma)

#callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
callbacks = []

varkeys_model.fit(x_train_pct, y_train_pct,
        batch_size=  batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks = callbacks)


scores = varkeys_model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (varkeys_model.metrics_names[1], scores[1]*100))

print(x_train.shape)