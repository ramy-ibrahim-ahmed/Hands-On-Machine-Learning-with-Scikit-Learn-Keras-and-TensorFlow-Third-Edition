import math
import matplotlib.pyplot as plt
from base import *
from keras import optimizers


### POWER SCHEDULING ###
"""
staircase=False:
    lr = lr0 / (1 + decay_rate * step / decay_steps)

staircase=True:
    lr = lr0 / (1 + decay_rate * step // decay_steps)
"""
power_scheduler = optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.01,
    decay_steps=10_000,
    decay_rate=1.0,
    staircase=False,
)
optimizer = optimizers.SGD(learning_rate=power_scheduler)
# history_power = build_net(optimizer=optimizer)


### PLOT ###
# initial_learning_rate = 0.01
# decay_rate = 1.0
# decay_steps = 10_000
# steps = np.arange(100_000)
# lrs = initial_learning_rate / (1 + decay_rate * steps / decay_steps)
# lrs2 = initial_learning_rate / (1 + decay_rate * steps // decay_steps)
# plt.plot(steps, lrs, "-", label="staircase=False")
# plt.plot(steps, lrs2, "-", label="staircase=True")
# plt.axis([0, steps.max(), 0, 0.0105])
# plt.xlabel("Step")
# plt.ylabel("Learning Rate")
# plt.title("Invert time decay scheduling", fontsize=20)
# plt.legend()
# plt.grid(True, lw=0.5, linestyle="--")
# plt.tight_layout()
# plt.show()


### EXPONENTIAL SCHEDULING ###
"""
lr = lr0 * decay_rate ** (step / decay_steps)
"""
exponential_scheduler = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=20_000,
    decay_rate=0.1,
    staircase=False,
)
optimizer = optimizers.SGD(learning_rate=exponential_scheduler)
# history_exponential = build_net(optimizer=optimizer)


### PLOT ###
# initial_learning_rate = 0.01
# decay_rate = 0.1
# decay_steps = 20_000
# steps = np.arange(100_000)
# lrs = initial_learning_rate * decay_rate ** (steps / decay_steps)
# lrs2 = initial_learning_rate * decay_rate ** np.floor(steps / decay_steps)
# plt.plot(steps, lrs, "-", label="staircase=False")
# plt.plot(steps, lrs2, "-", label="staircase=True")
# plt.axis([0, steps.max(), 0, 0.0105])
# plt.xlabel("Step")
# plt.ylabel("Learning Rate")
# plt.title("Exponential Scheduling", fontsize=20)
# plt.legend()
# plt.grid(True, lw=0.5, linestyle="--")
# plt.tight_layout()
# plt.show()


### CALLBACK FOR EXPONENTIAL SCHEDULING FOR EPOCH LVL ###
def exponential_decay(lr0, s):
    def equation(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return equation


ex_decay = exponential_decay(lr0=0.01, s=20)
schedular_callback = keras.callbacks.LearningRateScheduler(ex_decay, verbose=1)

NET = architecture()
NET.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    optimizer=optimizers.SGD(),
    metrics=["accuracy"],
)
# history = NET.fit(
#     xtrain,
#     ytrain,
#     epochs=10,
#     validation_data=(xtrain, ytrain),
#     callbacks=[schedular_callback],
# )


### Custom learnin rate sceduling ###
class ExponentialDecay(keras.callbacks.Callback):
    def __init__(self, n_steps, decay_base=0.1, initial_lr=0.01):
        super().__init__()
        if n_steps <= 0:
            raise ValueError("n_steps must be a positive integer.")
        self.n_steps = n_steps
        self.decay_base = decay_base
        self.initial_lr = initial_lr

    def on_batch_begin(self, batch, logs=None):
        step = self.model.optimizer.iterations
        new_lr = self.initial_lr * self.decay_base ** (step / self.n_steps)
        self.model.optimizer.learning_rate = new_lr
        return super().on_batch_begin(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = self.model.optimizer.learning_rate
        return super().on_epoch_end(epoch, logs)


lr0 = 0.1
batch_size = 32
n_epochs = 10
n_steps = n_epochs * math.ceil(len(xtrain) / batch_size)
schedular_callback = ExponentialDecay(n_steps, decay_base=0.1, initial_lr=lr0)
optimizer = keras.optimizers.SGD()

NET = architecture()
NET.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
# history = NET.fit(
#     xtrain,
#     ytrain,
#     epochs=n_epochs,
#     validation_data=(xtrain, ytrain),
#     callbacks=[schedular_callback],
# )


### PIECEWISE CONSTANT ###
# constant lr for a period
# 0 -> 50_000 | 50_000 -> 80_000 | 80_000 -> -1
lr_scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[50_000, 80_000],
    values=[0.01, 0.005, 0.001],
)
optimizer = keras.optimizers.SGD(learning_rate=lr_scheduler)
# history = build_net(optimizer=optimizer)


### Customization ###
def piecewise_constant_(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001


# common approutch
def piecewise_constant(boundaries, values):
    boundaries = np.array([0] + boundaries)
    values = np.array(values)

    def get_lr(epoch):
        return values[(boundaries > epoch).argmax() - 1]

    return get_lr


boundaries = [5, 15]
values = [0.01, 0.005, 0.001]
# pc = piecewise_constant(boundaries, values)
# lr_scheduler = keras.callbacks.LearningRateScheduler(pc)

lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_, verbose=1)

net = architecture()
optimizer = keras.optimizers.Nadam()
net.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
# history = net.fit(
#     xtrain,
#     ytrain,
#     epochs=25,
#     validation_data=(xvalid, yvalid),
#     callbacks=[lr_scheduler],
# )


### Preformance scheduling ###
# like early stopping when error stop dropping
# it decrease lr
net = architecture()
keras.utils.set_random_seed(42)
optimizer = keras.optimizers.SGD(learning_rate=0.01)
net.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=5,
)
# history = net.fit(
#     xtrain,
#     ytrain,
#     epochs=25,
#     validation_data=(xvalid, yvalid),
#     callbacks=[lr_scheduler],
# )


### PLOT ###
# plt.plot(history.epoch, history.history["learning_rate"], "bo-")
# plt.xlabel("Epoch")
# plt.ylabel("Learning Rate", color='b')
# plt.tick_params('y', colors='b')
# plt.gca().set_xlim(0, 25 - 1)
# plt.grid(True, lw=0.5, linestyle="--")
# ax2 = plt.gca().twinx()
# ax2.plot(history.epoch, history.history["val_loss"], "r^-")
# ax2.set_ylabel('Validation Loss', color='r')
# ax2.tick_params('y', colors='r')
# plt.title("Reduce learning rate on plateau", fontsize=20)
# plt.show()


### TRACK LOSS AND LEARNING RATE OVER 1 EPOCH ###
class ExponentioalLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.sum_epoch_loss = 0
        return super().on_epoch_begin(epoch, logs)

    def on_batch_end(self, batch, logs=None):
        mean_epoch_loss = logs["loss"]
        new_sum_epoch_loss = mean_epoch_loss * (batch + 1)
        batch_loss = new_sum_epoch_loss - self.sum_epoch_loss
        self.sum_epoch_loss = new_sum_epoch_loss
        self.rates.append(self.model.optimizer.learning_rate.numpy())
        self.losses.append(batch_loss)
        self.model.optimizer.learning_rate *= self.factor
        return super().on_batch_end(batch, logs)


def find_lr(model, x, y, epochs=1, batch_size=32, min_rate=1e-4, max_rate=0.5):
    init_W = model.get_weights()
    total_update_iterations = math.ceil(len(x) / batch_size) * epochs
    factor = (max_rate / min_rate) ** (1 / total_update_iterations)
    init_lr = model.optimizer.learning_rate
    model.optimizer.learning_rate = min_rate
    exp_lr = ExponentioalLearningRate(factor)
    history = model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[exp_lr],
    )
    model.optimizer.learning_rate = init_lr
    model.set_weights(init_W)
    return exp_lr.rates, exp_lr.losses


def plot_lr_vs_loss(rates, losses):
    plt.figure(figsize=(10, 6))
    plt.plot(rates, losses, "b.-", label="Loss")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate vs. Loss", fontsize=20)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    min_loss_idx = np.argmin(losses)
    min_loss = losses[min_loss_idx]
    min_rate = rates[min_loss_idx]
    plt.scatter(
        [min_rate],
        [min_loss],
        color="red",
        label=f"Min Loss: {min_loss:.4f} at LR: {min_rate:.4f}",
    )
    plt.legend()
    plt.show()


model = architecture()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"],
)
batch_size = 128
# rates, losses_ = find_lr(
#     model,
#     xtrain,
#     ytrain,
#     epochs=1,
#     batch_size=batch_size,
# )
# print(f"Initial lr = {rates[np.argmin(np.array(losses_))]}")
# plot_lr_vs_loss(rates, losses_)


### ONE CYCLE SCHEDULING ###
# Warm-up Phase: lr increases from start_lr to max_lr.
# Cool-down Phase: lr decreases from max_lr to start_lr.
# Final Phase: lr decreases further from start_lr to last_lr.
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(
        self,
        iterations,
        max_lr=1e-3,
        start_lr=None,
        last_iterations=None,
        last_lr=None,
    ):
        self.iterations = iterations
        self.max_lr = max_lr
        self.start_lr = start_lr or max_lr / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_lr = last_lr or self.start_lr / 1000
        self.iteration = 0

    def interpolate_(self, iter1, iter2, lr1, lr2):
        return (lr2 - lr1) * (self.iteration - iter1) / (iter2 - iter1) + lr1

    def on_batch_begin(self, batch, logs=None):
        if self.iteration < self.half_iteration:
            lr = self.interpolate_(
                0,
                self.half_iteration,
                self.start_lr,
                self.max_lr,
            )
        elif self.iteration < 2 * self.half_iteration:
            lr = self.interpolate_(
                self.half_iteration,
                2 * self.half_iteration,
                self.max_lr,
                self.start_lr,
            )
        else:
            lr = self.interpolate_(
                2 * self.half_iteration,
                self.iterations,
                self.start_lr,
                self.last_lr,
            )
        self.iteration += 1
        self.model.optimizer.learning_rate = lr
        return super().on_batch_begin(batch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        logs["learning_rate"] = self.model.optimizer.learning_rate
        return super().on_epoch_end(epoch, logs)
    
    # def on_epoch_begin(self, epoch, logs=None):
    #     print(f"learning_rate: {self.model.optimizer.learning_rate.numpy()}")
    #     return super().on_epoch_begin(epoch, logs)

    def on_train_begin(self, logs=None):
        self.iteration = 0
        return super().on_train_begin(logs)


net = architecture()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(),
    metrics=["accuracy"],
)
n_epochs = 25
iterations = math.ceil(len(xtrain) / batch_size) * n_epochs
onecycle = OneCycleScheduler(iterations, max_lr=0.1)
# history = model.fit(
#     xtrain,
#     ytrain,
#     epochs=n_epochs,
#     batch_size=batch_size,
#     validation_data=(xvalid, yvalid),
#     callbacks=[onecycle],
# )