import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd

keras.utils.set_random_seed(67)

### DATASET ###
fashion_mnist = keras.datasets.fashion_mnist.load_data()
(xtemp, ytemp), (xtest, ytest) = fashion_mnist
xtrain, ytrain = xtemp[:-5000], ytemp[:-5000]
xdev, ydev = xtemp[-5000:], ytemp[-5000:]
del xtemp, ytemp

# print(xtrain.shape)
# print(xtrain.dtype)

xtrain, xdev, xtest = xtrain / 255.0, xdev / 255.0, xtest / 255.0

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

SHOW = False
if SHOW:
    plt.imshow(xtrain[0], cmap="binary")
    plt.axis("off")
    plt.title(f"{class_names[ytrain[0]]}")
    plt.show()

SHOW = False
if SHOW:
    rows = 4
    cols = 10
    plt.figure(figsize=(cols * 1.2, rows * 1.2))
    for row in range(rows):
        for col in range(cols):
            index = cols * row + col
            plt.subplot(rows, cols, index + 1)
            plt.imshow(xtrain[index], cmap="binary", interpolation="nearest")
            plt.axis("off")
            plt.title(class_names[ytrain[index]])
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()


### NETWORK ###
from keras import Sequential
from keras.layers import InputLayer, Flatten, Dense
from keras.activations import relu, softmax

NET = Sequential(
    [
        InputLayer(shape=[28, 28]),
        Flatten(),
        Dense(units=300, activation=relu),
        Dense(units=100, activation=relu),
        Dense(units=10, activation=softmax),
    ]
)
# print(NET.summary())

# reset session and names for layers
# keras.backend.clear_session()

hidden1 = NET.layers[1]
# print(NET.layers)
# print(hidden1.name)
# print(NET.get_layer("dense") is hidden1)

weights_1, biases_1 = hidden1.get_weights()
# print(weights_1.shape)


### COMPILING ###
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD
from keras.metrics import sparse_categorical_accuracy

NET.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=SGD(learning_rate=0.01),  # default
    metrics=[sparse_categorical_accuracy],
)

# one-hot encoding
ytemp = ytrain[5:10]
y_01 = keras.utils.to_categorical(ytemp, num_classes=10)
# print(y_01)

# sparce encoding (one-hot decoding)
y_ = np.argmax(y_01, axis=1)
# print(y_)


### TRAINING ###
# HISTORY = NET.fit(
    # xtrain,
    # ytrain,
    # epochs=30,
    # validation_data=(xdev, ydev),
    # validation_split=0.2,  # 20% of the training data will be used for validation
# )
# print(HISTORY.params)
# print(HISTORY.epoch)

# J = pd.DataFrame(HISTORY.history)
# J.plot(
#     xlim=[0, 29],
#     ylim=[0, 1],
#     grid=True,
#     xlabel="Epoch",
#     style=["r--", "r--", "b-", "b-"],
# )
# plt.axvline(0.5, linewidth=1, c="black", linestyle="--")
# plt.show()

NET.evaluate(xtest, ytest)
xnew = xtest[:3]
yhat = NET.predict(xnew)
# print(f"{yhat.round(2)}")
i = yhat.argmax(axis=1)
labels = np.array(class_names)
# print(labels[i])

# plt.figure(figsize=(7.2, 2.4))
# for index, image in enumerate(xnew):
#     plt.subplot(1, 3, index + 1)
#     plt.imshow(image, cmap="binary", interpolation="nearest")
#     plt.axis('off')
#     plt.title(labels[index])
# plt.subplots_adjust(wspace=0.2, hspace=0.5)
# plt.show()

print(np.exp([-1, -2, -5]) - 1)