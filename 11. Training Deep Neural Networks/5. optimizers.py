from base import *
import keras
import matplotlib.pyplot as plt
from keras import layers, optimizers, losses


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


print("\nSGD\n")
optimizer = optimizers.SGD(
    learning_rate=0.001,
)
history_sgd = build_net(optimizer)

print("\nSGD with momentum\n")
optimizer = optimizers.SGD(
    learning_rate=0.001,
    momentum=0.9,
)
history_sgd_m = build_net(optimizer)

print("\nSGD with momentum & nesterov\n")
optimizer = optimizers.SGD(
    learning_rate=0.001,
    momentum=0.9,
    nesterov=True,
)
history_sgd_m_n = build_net(optimizer)

print("\nAdaGrad\n")
optimizer = optimizers.Adagrad(
    learning_rate=0.001,
)
history_adagrad = build_net(optimizer)

print("\nRmsProp\n")
optimizer = optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
)
history_rmsprop = build_net(optimizer)

print("\nAdam\n")
optimizer = optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
)
history_adam = build_net(optimizer)

print("\nAdamax\n")
optimizer = optimizers.Adamax(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
)
history_adamax = build_net(optimizer)

print("\nNAdam\n")
optimizer = optimizers.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
)
history_nadam = build_net(optimizer)

print("\nAdamW\n")
optimizer = optimizers.AdamW(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=1e-5,
)
history_adamw = build_net(optimizer)


title = "Optimizers in training data"
for loss in ("loss", "val_loss"):
    opt_names = "SGD Momentum Nesterov AdaGrad RMSProp Adam Adamax Nadam AdamW"
    for history, opt_name in zip(
        (
            history_sgd,
            history_sgd_m,
            history_sgd_m_n,
            history_adagrad,
            history_rmsprop,
            history_adam,
            history_adamax,
            history_nadam,
            history_adamw,
        ),
        opt_names.split(),
    ):
        plt.plot(history.history[loss], label=f"{opt_name}")
    plt.axis([0, 9, 0.1, 0.7])
    plt.ylabel({"loss": "Training loss", "val_loss": "Validation loss"}[loss])
    plt.xlabel("Epochs")
    plt.title(title, fontsize=20)
    title = "Optimizers in validation data"
    plt.legend(loc='lower left')
    plt.grid(True, lw=0.5, linestyle="--")
    plt.tight_layout()
    plt.show()