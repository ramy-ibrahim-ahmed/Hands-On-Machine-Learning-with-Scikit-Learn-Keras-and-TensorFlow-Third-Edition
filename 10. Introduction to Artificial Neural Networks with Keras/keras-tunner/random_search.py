import numpy
import keras
import keras_tuner
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, Adam
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


random_search_tuner = keras_tuner.RandomSearch(
    create_nn,
    objective="val_accuracy",
    max_trials=10,
    # overwrite=True,
    overwrite=False,
    directory="my_fashion_mnist",
    project_name="random_search",
    seed=67,
)

random_search_tuner.search(
    xtrain,
    ytrain,
    epochs=10,
    validation_data=(xdev, ydev),
)

top3_nets = random_search_tuner.get_best_models(num_models=3)
best_net = top3_nets[0]
print(best_net)

top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
print(top3_params[0].values)

best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
best_trial.summary()

val_accuracy = best_trial.metrics.get_best_value("val_accuracy")
print(val_accuracy)

best_net.fit(xtrain, ytrain, epochs=10)
test_loss, test_accuracy = best_net.evaluate(xtest, ytest)
print(f"Test accuarcy = {test_accuracy}")