import keras
from base import *

# Set random seed for reproducibility
keras.utils.set_random_seed(42)

### BASE NET ###
NETA = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        keras.layers.Dense(8, activation="softmax"),
    ]
)
NETA.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"],
)

# Fit the model
# history = NETA.fit(
#     xtrain_A,
#     ytrain_A,
#     epochs=20,
#     validation_data=(xdev_A, ydev_A),
# )

# Save the model
# NETA.save("11. Training Deep Neural Networks/NETA.keras")

### NET B ###
# Load model
NETA_ = keras.models.load_model(r"11. Training Deep Neural Networks\NETA.keras")

# Clone the model
NETA_clone = keras.models.clone_model(NETA_)
NETA_clone.set_weights(NETA_.get_weights())

# Create a new model by reusing layers from the cloned model
NETB = keras.Sequential(NETA_clone.layers[:-1])
NETB.add(keras.layers.Dense(1, activation="sigmoid"))

### FREEZING ###
for layer in NETB.layers[:-1]:
    layer.trainable = False

# Compile the model after freezing layers
optimizer = keras.optimizers.SGD(learning_rate=0.001)
NETB.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)

# Train the model with frozen layers
history = NETB.fit(
    xtrain_B,
    ytrain_B,
    epochs=4,
    validation_data=(xdev_B, ydev_B),
)

### UNFREEZING ###
for layer in NETB.layers[:-1]:
    layer.trainable = True

# Compile the model after unfreezing layers
optimizer = keras.optimizers.SGD(learning_rate=0.001)
NETB.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)

# Train the model with unfrozen layers
history = NETB.fit(
    xtrain_B,
    ytrain_B,
    epochs=16,
    validation_data=(xdev_B, ydev_B),
)

# Evaluate the model
print(NETB.evaluate(xtest_B, ytest_B))