import numpy
import keras
from keras import Sequential, layers

numpy.random.seed(67)
keras.utils.set_random_seed(67)

### DATASET ###
fashion_mnist = keras.datasets.fashion_mnist.load_data()
(XTRAIN, YTRAIN), (xtest, ytest) = fashion_mnist
xtrain, ytrain = XTRAIN[:-5000], YTRAIN[:-5000]
xdev, ydev = XTRAIN[-5000:], YTRAIN[-5000:]
xtrain1, xdev1, xtest1 = xtrain / 255, xdev / 255, xtest / 255


### NO BN ###
NET1 = Sequential(
    [
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(10, activation="softmax"),
    ]
)
NET1.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"],
)
# NET1.fit(
#     xtrain1,
#     ytrain,
#     epochs=10,
#     validation_data=(xdev1, ydev),
# )


### BN AFTER ACTIVATION ###
NET2 = Sequential(
    [
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dense(10, activation="softmax"),
    ]
)
# NET2.summary()
for var in NET2.layers[1].variables:
    print(var.name, var.trainable)
NET2.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"],
)
# NET2.fit(
#     xtrain,
#     ytrain,
#     epochs=10,
#     validation_data=(xdev, ydev),
# )


### BN BEFORE ACTIVATION ###
NET3 = Sequential(
    [
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
NET3.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"],
)
# NET3.fit(
#     xtrain,
#     ytrain,
#     epochs=10,
#     validation_data=(xdev, ydev),
# )


### GRADIENT CLIPPING ### 93%
optimizer = keras.optimizers.SGD(clipvalue=1.0)
NET3.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
# NET3.fit(
#     xtrain,
#     ytrain,
#     epochs=10,
#     validation_data=(xdev, ydev),
# )


### NORM CLIPPING ### 90%
optimizer = keras.optimizers.SGD(clipnorm=1.0)
NET3.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
# NET3.fit(
#     xtrain,
#     ytrain,
#     epochs=10,
#     validation_data=(xdev, ydev),
# )