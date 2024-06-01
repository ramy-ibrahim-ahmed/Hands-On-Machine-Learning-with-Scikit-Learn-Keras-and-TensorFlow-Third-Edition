import numpy as np
import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

np.random.seed(67)
keras.utils.set_random_seed(67)

### DATASET ###
housing = fetch_california_housing()
xtrain, xtemp, ytrain, ytemp = train_test_split(
    housing.data, housing.target, test_size=0.20
)
xdev, xtest, ydev, ytest = train_test_split(xtemp, ytemp, test_size=0.50)
del xtemp, ytemp


### NETWORK ###
from keras import Sequential
from keras.layers import Dense, InputLayer, Normalization
from keras.activations import relu
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.metrics import RootMeanSquaredError

NORM_LAYER = Normalization()
NET = Sequential(
    [
        InputLayer(shape=xtrain.shape[1:]),
        NORM_LAYER,
        Dense(units=50, activation=relu),
        Dense(units=50, activation=relu),
        Dense(units=50, activation=relu),
        Dense(units=1),
    ]
)

NET.compile(
    loss=mean_squared_error,
    optimizer=Adam(1e-3),
    metrics=[RootMeanSquaredError],
)

NET.summary()
print(xtrain.shape)

NORM_LAYER.adapt(xtrain)

HISTORY = NET.fit(
    xtrain,
    ytrain,
    epochs=20,
    validation_data=(xdev, ydev),
)

# J, SCORE = NET.evaluate(xtrain, ytrain)
# print(f"mse = {J}")
# print(f"rmse = {SCORE}")

# xnew = xtest[:3]
# yhat = NET.predict(xnew)
# print(yhat)