import numpy
import keras
from keras import Sequential, layers
import time

numpy.random.seed(67)
keras.utils.set_random_seed(67)

fashion_mnist = keras.datasets.fashion_mnist.load_data()
(XTRAIN, YTRAIN), (xtest, ytest) = fashion_mnist
xtrain, ytrain = XTRAIN[:-5000], YTRAIN[:-5000]
xdev, ydev = XTRAIN[-5000:], YTRAIN[-5000:]
xtrain, xdev, xtest = xtrain / 255, xdev / 255, xtest / 255

pixel_mean = xtrain.mean(axis=0, keepdims=True)
pixel_std = xtrain.std(axis=0, keepdims=True)
xtrain_scaled = (xtrain - pixel_mean) / pixel_std
xdev_scaled = (xdev - pixel_mean) / pixel_std
xtest_scaled = (xtest - pixel_mean) / pixel_std


### RELU ###
NET_RELU = Sequential()
NET_RELU.add(layers.Input(shape=(28, 28)))
NET_RELU.add(layers.Flatten())
for layer in range(100):
    NET_RELU.add(
        layers.Dense(
            units=100,
            activation="relu",
            kernel_initializer="he_normal",
        )
    )
NET_RELU.add(layers.Dense(units=10, activation="softmax"))
NET_RELU.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"],
)
tic = time.time()
HISTORY = NET_RELU.fit(
    xtrain_scaled,
    ytrain,
    epochs=5,
    validation_data=(xdev_scaled, ydev),
)
toc = time.time()
RELU_TIME = toc - tic


### LEAKY RELU ###
NET_LEAKY_RELU = Sequential()
NET_LEAKY_RELU.add(layers.Input(shape=(28, 28)))
NET_LEAKY_RELU.add(layers.Flatten())
for layer in range(100):
    NET_LEAKY_RELU.add(
        layers.Dense(
            units=100,
            activation="leaky_relu",
            kernel_initializer="he_normal",
        )
    )
NET_LEAKY_RELU.add(layers.Dense(units=10, activation="softmax"))
NET_LEAKY_RELU.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"],
)
tic = time.time()
HISTORY = NET_LEAKY_RELU.fit(
    xtrain_scaled,
    ytrain,
    epochs=5,
    validation_data=(xdev_scaled, ydev),
)
toc = time.time()
LEAKY_RELU_TIME = toc - tic


### ELU ###
NET_ELU = Sequential()
NET_ELU.add(layers.Input(shape=(28, 28)))
NET_ELU.add(layers.Flatten())
for layer in range(100):
    NET_ELU.add(
        layers.Dense(
            units=100,
            activation="elu",
            kernel_initializer="he_normal",
        )
    )
NET_ELU.add(layers.Dense(units=10, activation="softmax"))
NET_ELU.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"],
)
tic = time.time()
HISTORY = NET_ELU.fit(
    xtrain_scaled,
    ytrain,
    epochs=5,
    validation_data=(xdev_scaled, ydev),
)
toc = time.time()
ELU_TIME = toc - tic


### SELU ###
NET_SELU = Sequential()
NET_SELU.add(layers.Input(shape=(28, 28)))
NET_SELU.add(layers.Flatten())
for layer in range(100):
    NET_SELU.add(
        layers.Dense(
            units=100,
            activation="selu",
            kernel_initializer="lecun_normal",
        )
    )
NET_SELU.add(layers.Dense(units=10, activation="softmax"))
NET_SELU.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"],
)
tic = time.time()
HISTORY = NET_SELU.fit(
    xtrain_scaled,
    ytrain,
    epochs=5,
    validation_data=(xdev_scaled, ydev),
)
toc = time.time()
SELU_TIME = toc - tic


_, RELU_DEV = NET_RELU.evaluate(xdev_scaled, ydev)
_, RELU_TEST = NET_RELU.evaluate(xtest_scaled, ytest)
print(f"RELU dev = {RELU_DEV}, test = {RELU_TEST}, time = {RELU_TIME}")

_, LEAKY_RELU_DEV = NET_LEAKY_RELU.evaluate(xdev_scaled, ydev)
_, LEAKY_RELU_TEST = NET_LEAKY_RELU.evaluate(xtest_scaled, ytest)
print(f"LEAKY RELU dev = {LEAKY_RELU_DEV}, test = {LEAKY_RELU_TEST}, time = {LEAKY_RELU_TIME}")

_, ELU_DEV = NET_ELU.evaluate(xdev_scaled, ydev)
_, ELU_TEST = NET_ELU.evaluate(xtest_scaled, ytest)
print(f"ELU RELU dev = {ELU_DEV}, test = {ELU_TEST}, time = {ELU_TIME}")

_, SELU_DEV = NET_SELU.evaluate(xdev_scaled, ydev)
_, SELU_TEST = NET_SELU.evaluate(xtest_scaled, ytest)
print(f"SELU dev = {SELU_DEV}, test = {SELU_TEST}, time = {SELU_TIME}")