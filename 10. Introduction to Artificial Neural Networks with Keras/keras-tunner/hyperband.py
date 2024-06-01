import numpy
import keras
import keras_tuner
from pathlib import Path
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Normalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split

numpy.random.seed(67)
keras.utils.set_random_seed(67)

(XTRAIN, YTRAIN), (xtest, ytest) = fashion_mnist.load_data()
xtrain, xdev, ytrain, ydev = train_test_split(XTRAIN, YTRAIN, test_size=0.10, random_state=67)

def create_nn(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        SGD(learning_rate=learning_rate)
    else:
        Adam(learning_rate=learning_rate)
    
    NET = Sequential()
    NET.add(Flatten())
    for _ in range(n_hidden):
        NET.add(Dense(units=n_neurons, activation="relu"))
    NET.add(Dense(units=10, activation="softmax"))
    
    NET.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    return NET


class MyClassificationHyperNet(keras_tuner.HyperModel):
    def build(self, hp):
        return create_nn(hp)
    
    def fit(self, hp, NET, x, y, **kwargs):
        if hp.Boolean("normalize"):
            NORM_LAYER = Normalization()
            x = NORM_LAYER(x)
        return NET.fit(x, y, **kwargs)


### Hyperband ###
hyperband_tuner = keras_tuner.Hyperband(
    MyClassificationHyperNet(),
    objective="val_accuracy",
    seed=67,
    max_epochs=10,
    factor=3,
    hyperband_iterations=2,
    overwrite=False,
    directory="my_fashion_mnist_hyperband",
    project_name="hyperband",
)

root_logdir = Path(hyperband_tuner.directory) / "tensorboard"
tensorboard_cb = TensorBoard(root_logdir)
early_stopping_cb = EarlyStopping(monitor="val_accuracy", mode="maximize", patience=2)

hyperband_tuner.search(
    xtrain,
    ytrain,
    epochs=10,
    validation_data=(xdev, ydev),
    callbacks=[early_stopping_cb, tensorboard_cb],
)