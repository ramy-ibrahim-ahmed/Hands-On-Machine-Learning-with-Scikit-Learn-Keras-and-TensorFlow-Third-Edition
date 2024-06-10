from base import *
from functools import partial

keras.utils.set_random_seed(42)

### L1 & L2 regularization ###
RegularizedDense = partial(
    layers.Dense,
    activation="relu",
    kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2(0.01),
)

net = keras.Sequential(
    [
        keras.Input(shape=[28, 28]),
        layers.Flatten(),
        RegularizedDense(100),
        RegularizedDense(100),
        RegularizedDense(10, activation="softmax"),
    ]
)
optimizer = keras.optimizers.SGD(learning_rate=0.02)
net.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
# history = net.fit(
#     xtrain,
#     ytrain,
#     epochs=10,
#     validation_data=(xvalid, yvalid),
# )


### DopOut ###
net = keras.Sequential(
    [
        keras.Input(shape=[28, 28]),
        layers.Flatten(),
        layers.Dropout(rate=0.2),
        layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(rate=0.2),
        layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(rate=0.2),
        layers.Dense(10, activation="softmax"),
    ]
)
optimizer = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
)
net.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
# print(net.evaluate(xtrain, ytrain))
# print(net.evaluate(xtest, ytest))


### MC DropOut ###
y_probas = np.stack([net(xtest, training=True) for _ in range(100)])
y_proba = y_probas.mean(axis=0)
y_std = y_probas.std(axis=0)

# print(net.predict(xtest[:1]).round(3))
# print(y_proba[0].round(3))
# print(y_std[0].round(3))

y_pred = y_proba.argmax(axis=1)
accuracy = (y_pred == ytest).sum() / len(ytest)
# print(accuracy)


### MCDropout layer ###
class MCDropout(layers.Dropout):
    def call(self, inputs, training=False):
        training = True
        return super().call(inputs, training)


Dropout = layers.Dropout
mc_net = keras.Sequential(
    [
        MCDropout(layer.rate) if isinstance(layer, Dropout) else layer
        for layer in net.layers
    ]
)
# mc_net.set_weights(net.get_weights())
# mc_net.summary()

# inference
# mc_pred = np.mean([mc_net.predict(xtest[:1]) for _ in range(100)], axis=0).round(2)
# print(mc_pred)


### Max Norm ###
MaxnormDense = partial(
    layers.Dense,
    activation="relu",
    kernel_initializer="he_normal",
    kernel_constraint=keras.constraints.max_norm(1.0),
)

net = keras.Sequential(
    [
        keras.Input(shape=[28, 28]),
        layers.Flatten(input_shape=[28, 28]),
        MaxnormDense(100),
        MaxnormDense(100),
        layers.Dense(10, activation="softmax"),
    ]
)
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
net.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
# history = net.fit(
#     xtrain,
#     ytrain,
#     epochs=10,
#     validation_data=(xvalid, yvalid),
# )