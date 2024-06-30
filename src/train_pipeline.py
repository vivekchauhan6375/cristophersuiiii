import tensorflow as tf
import numpy as np
from src.config.config import epochs
from src.pipeline import functional_nn
from src.preprocessing.preprocessors import optimizer
from src.preprocessing.preprocessors import binary_cross_entropy_loss
from src.preprocessing.data_management import save_model
from src.config import config as c
from src.config.config import X_train, Y_train, mb_size

def training_data_generator():
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]

    for i in range(n_samples // mb_size):
        start_index = i * mb_size
        end_index = start_index + mb_size
        X_train_mb = X_train_shuffled[start_index:end_index]
        Y_train_mb = Y_train_shuffled[start_index:end_index]
        yield X_train_mb, Y_train_mb


for e in range(epochs=100):

    for X_train_mb, Y_train_mb in training_data_generator():
        with tf.GradientTape() as tape:
            Y_pred = functional_nn(X_train_mb, training=True)  
            loss_func = binary_cross_entropy_loss(Y_pred, Y_train_mb)

        gradients = tape.gradient(loss_func, functional_nn.trainable_weights)
        optimizer.apply_gradients(zip(gradients, functional_nn.trainable_weights))

                
    print("Epoch # {}, Loss = {}".format(e + 1, loss_func))

# Save the model
if __name__ == "__main__":
    save_model(functional_nn, c)