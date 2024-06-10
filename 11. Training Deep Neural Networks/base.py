import keras
from keras import layers, losses
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist.load_data()
(xtrain_full, ytrain_full), (xtest, ytest) = fashion_mnist
xtrain, ytrain = xtrain_full[:-5000], ytrain_full[:-5000]
xvalid, yvalid = xtrain_full[-5000:], ytrain_full[-5000:]
xtrain, xvalid, xtest = xtrain / 255, xvalid / 255, xtest / 255
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
pos_class_id = class_names.index("Pullover")
neg_class_id = class_names.index("T-shirt/top")


def split_dataset(X, y):
    y_for_B = (y == pos_class_id) | (y == neg_class_id)
    y_A = y[~y_for_B]
    y_B = (y[y_for_B] == pos_class_id).astype(np.float32)
    old_class_ids = list(set(range(10)) - set([neg_class_id, pos_class_id]))
    for old_class_id, new_class_id in zip(old_class_ids, range(8)):
        y_A[y_A == old_class_id] = new_class_id  # reorder class ids for A
    return ((X[~y_for_B], y_A), (X[y_for_B], y_B))


(xtrain_A, ytrain_A), (xtrain_B, ytrain_B) = split_dataset(xtrain, ytrain)
(xdev_A, ydev_A), (xdev_B, ydev_B) = split_dataset(xvalid, yvalid)
(xtest_A, ytest_A), (xtest_B, ytest_B) = split_dataset(xtest, ytest)
xtrain_B = xtrain_B[:200]
ytrain_B = ytrain_B[:200]


def architecture(seed=67):
    keras.utils.set_random_seed(seed=seed)
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(10, activation="softmax"),
        ]
    )


def build_net(optimizer):
    NET = architecture()
    NET.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return NET.fit(
        xtrain,
        ytrain,
        epochs=10,
        validation_data=(xvalid, yvalid),
    )