import numpy as np
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier

np.random.seed(42)

### DATASET REGRESSION ###
data = fetch_california_housing()
xtrain, xtemp, ytrain, ytemp = train_test_split(data.data, data.target, test_size=0.40)
xdev, xtest, ydev, ytest = train_test_split(xtemp, ytemp, test_size=0.50)

### MLPRegressor ###
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50])
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(xtrain, ytrain)
yhat = pipeline.predict(xdev)
Jdev = mean_squared_error(ydev, yhat, squared=False)
print(f"MSE in development for Mlpregressior = {Jdev}")


### DATASET CLASSIFICATION ###
data = load_iris()
xtrain, xtemp, ytrain, ytemp = train_test_split(data.data, data.target, test_size=0.20)
xdev, xtest, ydev, ytest = train_test_split(xtemp, ytemp, test_size=0.50)

### MLPClassifier ###
mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000)
pipline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(xtrain, ytrain)
accuricy = pipeline.score(xdev, ydev)
print(f"MLPClassifier accuricy = {accuricy}")