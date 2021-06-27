import tensorflow as tf
from tensorflow import keras
import numpy as np
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_labels/255.0
test_images = test_images/255.0
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy')

model.fit(test_images, test_labels, epochs=500)

label_real = test_labels[505]
input_real = test_images[505, :, :]
input_real_2 = np.zeros((1, 28, 28))
input_real_2[0, :, : ] = input_real

label_prediction = np.argmax(model.predict(input_real_2))
print("real_label: ", label_real)
print("predicted_label: ", label_prediction)


