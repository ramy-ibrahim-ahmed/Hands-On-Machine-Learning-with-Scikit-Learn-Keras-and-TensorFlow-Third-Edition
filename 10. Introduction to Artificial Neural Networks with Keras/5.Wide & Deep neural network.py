import numpy as np
import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Input, Normalization, Dense, Concatenate, concatenate
from keras.activations import relu

np.random.seed(67)
keras.utils.set_random_seed(67)

### DATASET ###
housing = fetch_california_housing()
xtrain, xtemp, ytrain, ytemp = train_test_split(
    housing.data, housing.target, test_size=0.20
)
xdev, xtest, ydev, ytest = train_test_split(xtemp, ytemp, test_size=0.50)
del xtemp, ytemp


### LAYERS ###
INPUT_LAYER = Input(shape=xtrain.shape[1:])
NORM_LAYER = Normalization()
HIDDEN_LAYER_1 = Dense(units=30, activation=relu)
HIDDEN_LAYER_2 = Dense(units=30, activation=relu)
CONCAT_LAYER = Concatenate()
OUTPUT_LAYER = Dense(units=1)


### DESIGN ###
input_ = INPUT_LAYER
normalized = NORM_LAYER(input_)
hidden1 = HIDDEN_LAYER_1(normalized)
hidden2 = HIDDEN_LAYER_2(hidden1)
concat = CONCAT_LAYER([normalized, hidden2])
output = OUTPUT_LAYER(concat)

WIDE_DEEP_NET_1 = Model(inputs=[input_], outputs=[output])

# WIDE_DEEP_NET_1.summary()

# WIDE_DEEP_NET_1.compile(
#     loss="mse",
#     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#     metrics=[keras.metrics.RootMeanSquaredError],
# )

# NORM_LAYER.adapt(xtrain)

# H = WIDE_DEEP_NET_1.fit(
#     xtrain,
#     ytrain,
#     epochs=20,
#     validation_data=(xdev, ydev)
# )

# J, _ = WIDE_DEEP_NET_1.evaluate(xtest, ytest)
# print(f"J = {J}")


keras.backend.clear_session()
keras.utils.set_random_seed(67)

### DESIGN ###
input_wide = Input(shape=[5])  # feature 0 -> 4
input_deep = Input(shape=[6])  # feature 2 -> 7
NORM_LAYER_WIDE = Normalization()
NORM_LAYER_DEEP = Normalization()
norm_wide = NORM_LAYER_WIDE(input_wide)
norm_deep = NORM_LAYER_DEEP(input_deep)
hidden1 = Dense(units=30, activation=relu)(norm_deep)
hidden2 = Dense(units=30, activation=relu)(hidden1)
concat = concatenate([norm_wide, hidden2])
output = Dense(units=1)(concat)

WIDE_DEEP_NET_2 = Model(
    inputs={"input_wide": input_wide, "input_deep": input_deep},
    outputs=[output],
)

# WIDE_DEEP_NET_2.compile(
#     loss="mse",
#     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#     metrics=[keras.metrics.RootMeanSquaredError],
# )

xtrain_wide, xtrain_deep = xtrain[:, :5], xtrain[:, 2:]
xdev_wide, xdev_deep = xdev[:, :5], xdev[:, 2:]
xtest_wide, xtest_deep = xtest[:, :5], xtest[:, 2:]
xnew_wide, xnew_deep = xtest_wide[:3], xtest_deep[:3]

# NORM_LAYER_WIDE.adapt(xtest_wide)
# NORM_LAYER_DEEP.adapt(xtest_deep)

# H = WIDE_DEEP_NET_2.fit(
#     {"input_wide": xtrain_wide, "input_deep": xtrain_deep},
#     ytrain,
#     epochs=20,
#     validation_data=({"input_wide": xdev_wide, "input_deep": xdev_deep}, ydev),
# )

# J, _ = WIDE_DEEP_NET_2.evaluate(
#     {"input_wide": xtest_wide, "input_deep": xtest_deep},
#     ytest,
# )
# print(f"J = {J}")

# yhat = WIDE_DEEP_NET_2.predict({"input_wide": xnew_wide, "input_deep": xnew_deep})
# print(f"yhat = {yhat}")


keras.backend.clear_session()
keras.utils.set_random_seed(67)

### DESIGN ###
input_wide = Input(shape=[5])
input_deep = Input(shape=[6])
NORM_LAYER_WIDE = Normalization()
NORM_LAYER_DEEP = Normalization()
norm_wide = NORM_LAYER_WIDE(input_wide)
norm_deep = NORM_LAYER_DEEP(input_deep)
hidden1 = Dense(units=30, activation=relu)(norm_deep)
hidden2 = Dense(units=30, activation=relu)(hidden1)
concat = concatenate(inputs=(norm_wide, hidden2))
auxiliary = Dense(units=1)(hidden2)
output = Dense(units=1)(concat)

WIDE_DEEP_NET_3 = Model(
    inputs=(input_wide, input_deep),
    outputs=(output, auxiliary),
)

# WIDE_DEEP_NET_3.compile(
#     loss=("mse", "mse"),
#     loss_weights=(0.9, 0.1),
#     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#     metrics=["RootMeanSquaredError", "RootMeanSquaredError"],
# )

# NORM_LAYER_WIDE.adapt(xtrain_wide)
# NORM_LAYER_DEEP.adapt(xtrain_deep)

# H = WIDE_DEEP_NET_3.fit(
#     (xtrain_wide, xtrain_deep),
#     (ytrain, ytrain),
#     epochs=20,
#     validation_data=((xdev_wide, xdev_deep), (ydev, ydev)),
# )

# J_SCORE = WIDE_DEEP_NET_3.evaluate((xtest_wide, xtest_deep), (ytest, ytest), return_dict=True)
# print(J_SCORE)

# YOUT, YAUX = WIDE_DEEP_NET_3.predict((xnew_wide, xnew_deep))
# print(f"YOUT = {YOUT}\n\n YAUX = {YAUX}")

# YHAT = WIDE_DEEP_NET_3.predict((xnew_wide, xnew_deep))
# YDICT = dict(zip(WIDE_DEEP_NET_3.output_names, YHAT))
# print(YDICT)


"""DYNAMIC API"""
keras.backend.clear_session()
keras.utils.set_random_seed(67)

### DESIGN ###
class WIDE_DEEP_NET_4(Model):
    def __init__(self, units=30, activation=relu, **kwargs):
        super().__init__(**kwargs)
        self.NORM_LAYER_WIDE = Normalization()
        self.NORM_LAYER_DEEP = Normalization()
        self.hidden1 = Dense(units=units, activation=activation)
        self.hidden2 = Dense(units=units, activation=activation)
        self.main_output = Dense(units=1)
        self.auxiliary = Dense(units=1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.NORM_LAYER_WIDE(input_wide)
        norm_deep = self.NORM_LAYER_DEEP(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = concatenate(inputs=(norm_wide, hidden2))
        auxiliary = self.auxiliary(hidden2)
        output = self.main_output(concat)
        return output, auxiliary


DYNAMIC_NET = WIDE_DEEP_NET_4(units=30, activation=relu, name="my_firist_cool_model")

DYNAMIC_NET.compile(
    loss="mse",
    loss_weights=[0.9, 0.1],
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["RootMeanSquaredError", "RootMeanSquaredError"],
)

DYNAMIC_NET.NORM_LAYER_WIDE.adapt(xtrain_wide)
DYNAMIC_NET.NORM_LAYER_DEEP.adapt(xtrain_deep)

# H = DYNAMIC_NET.fit(
#     (xtrain_wide, xtrain_deep),
#     (ytrain, ytrain),
#     epochs=10,
#     validation_data=((xdev_wide, xdev_deep), (ydev, ydev))
# )

# J_SCORE = DYNAMIC_NET.evaluate((xtest_wide, xtest_deep), (ytest, ytest), return_dict=True)
# print(J_SCORE)

# YHAT, YAUX = DYNAMIC_NET.predict((xnew_wide, xnew_deep))
# print(YHAT)


"""CALLBACKS"""
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    r"Part II. Neural Networks and Deep Learning\10. Introduction to Artificial Neural Networks with Keras\my_checkpoints_early_stopping.keras",
    save_best_only=True,
)

# H = DYNAMIC_NET.fit(
#     (xtrain_wide, xtrain_deep),
#     (ytrain, ytrain),
#     epochs=10,
#     validation_data=((xdev_wide, xdev_deep), (ydev, ydev)),
#     callbacks=[checkpoint_cb],
# )

early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

# H = DYNAMIC_NET.fit(
#     (xtrain_wide, xtrain_deep),
#     (ytrain, ytrain),
#     epochs=100,
#     validation_data=((xdev_wide, xdev_deep), (ydev, ydev)),
#     callbacks=[checkpoint_cb, early_stopping_cb],
# )


class DevTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")


val_train_ratio_cb = DevTrainRatioCallback()
# H = DYNAMIC_NET.fit(
#     (xtrain_wide, xtrain_deep),
#     (ytrain, ytrain),
#     epochs=10,
#     validation_data=((xdev_wide, xdev_deep), (ydev, ydev)),
#     callbacks=[val_train_ratio_cb],
#     verbose=0,
# )