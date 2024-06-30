import tensorflow as tf
from keras.optimizers import RMSprop #type: ignore

# Define the loss function
def binary_cross_entropy_loss(Y_hat, Y_true):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=Y_true, y_pred=Y_hat))


# Optimizer
optimizer = RMSprop(learning_rate=0.005)