import tensorflow as tf
from os import path, getcwd, chdir
path = "/home/jose/Documents/Professional Development/Tensorflow Artificial Intelligence/mnist.npz"

ACCURACY_THRESHOLD = 0.99

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy') > ACCURACY_THRESHOLD):
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
			self.model.stop_training = True


def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE


    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    x_train = x_train/255.0
    x_test = x_test/255.0

    callbacks = myCallback()
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit( x_train, y_train, epochs=10, callbacks=[callbacks])
    # model fitting
    hey = 0
    return history.epoch, history.history['accuracy'][-1]


train_mnist()