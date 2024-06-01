import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron, SGDClassifier
from scipy.special import expit as sigmoid

### DATASET ###
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target == 0


### PERCEPTRON ###
classifier_perc = Perceptron(random_state=42)
classifier_perc.fit(X, y)


### PREDICTIONS ###
xnew = [[2, 0.5], [3, 1]]
yhat = classifier_perc.predict(xnew)
print(yhat)


### SGDClassifier WITH SETTINGS LIKE PERCEPTRON ###
classifier_sgd = SGDClassifier(
    loss="perceptron",
    penalty=None,
    learning_rate="constant",
    eta0=1,
    random_state=42,
)
classifier_sgd.fit(X, y)


### VALIDATION ###
assert (classifier_sgd.coef_ == classifier_perc.coef_).all()
assert (classifier_sgd.intercept_ == classifier_perc.intercept_).all()


### RELU ###
def relu(z):
    return np.maximum(0, z)


### THE CENTER FINITE DIFFERENCE METHOD ###
def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps)) / (2 * eps)


max_z = 4.5
z = np.linspace(-max_z, max_z, 200)


### ACTIVATIONS ###
SHOW = False
if SHOW:
    plt.plot([-max_z, 0], [0, 0], "r-", linewidth=2, label="Heaviside")
    plt.plot([0, 0], [0, 1], "r-", linewidth=0.5)
    plt.plot([0, max_z], [1, 1], "r-", linewidth=2)
    plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
    plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
    plt.plot(z, np.tanh(z), "b-", linewidth=1, label="Tanh")
    plt.title("Activation functions", fontsize=16)
    plt.axis([-max_z, max_z, -1.65, 2.4])
    plt.gca().set_yticks([-1, 0, 1, 2])
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


### DERIVATIVES ###
SHOW = False
if SHOW:
    plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Heaviside")
    plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
    plt.plot(z, derivative(np.tanh, z), "b-", linewidth=1, label="Tanh")
    plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="Tanh")
    plt.plot(0, 0, "ro", markersize=5)
    plt.plot(0, 0, "rx", markersize=10)
    plt.plot(0, 1, "mo", markersize=5)
    plt.plot(0, 1, "mx", markersize=10)
    plt.title("Derivatives", fontsize=16)
    plt.axis([-max_z, max_z, -0.2, 1.2])
    plt.grid(True, linestyle="--", lw=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()