from keras.layers import Input, Dense #type: ignore
from keras.models import Model #type: ignore
from src.preprocessing.data_management import load_dataset
from src.config import config as c
from src.preprocessing.preprocessors import optimizer, binary_cross_entropy_loss


def initialize_data():
    data = load_dataset("train.csv")
    c.X_train = data.iloc[:, :-1].values 
    c.Y_train = data.iloc[:, -1].values.reshape(-1, 1)
    c.training_data = data


def functional_mlp():
    inp = Input(shape=(c.X_train.shape[1],))
    first_hidden_out = Dense(units=4, activation="relu")(inp)
    second_hidden_out = Dense(units=2, activation="relu")(first_hidden_out)
    nn_out = Dense(units=1, activation="sigmoid")(second_hidden_out)
    return Model(inputs=[inp], outputs=[nn_out])

initialize_data()


functional_nn = functional_mlp()

functional_nn.compile(optimizer=optimizer, loss=binary_cross_entropy_loss)

functional_nn.summary()