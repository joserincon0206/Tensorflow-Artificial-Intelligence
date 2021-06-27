import tensorflow as tf
import numpy as np
from tensorflow import keras

# Create 1 layer with 1 neuron input is shaped to just 1 value
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile Neural Network model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Providing Data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)


# Training the neural network
model.fit(xs, ys, epochs=1000, steps_per_epoch=5)

print(model.predict([10.0]))



