import tensorflow as tf
import numpy as np
from tensorflow import keras
# GRADED FUNCTION: house_model
def house_model(y_new):

    xs = np.arange(start=1, stop=11, dtype= float)
    ys = np.arange(start=0, stop=10, dtype= float)*.5 + 1

    print(xs)
    print(ys)


    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    y_predicted = model.predict([y_new])[0][0]
    y_predicted2 = np.array([y_predicted])
    return y_predicted2

prediction = house_model(7.0)
print(prediction)

