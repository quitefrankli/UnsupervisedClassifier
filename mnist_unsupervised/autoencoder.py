from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import random

def main():
	# size of encoded representation, aka hidden layer
	encoding_dim = 10

	autoencoder = Sequential()
	autoencoder.add(Dense(encoding_dim, input_shape=(784,), activation='relu'))
	autoencoder.add(Dense(784, activation='sigmoid'))

	# ----------------

	# train the autoencoder
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	(x_train, _), (x_test, y_test) = mnist.load_data()

	# normalise input data
	x_train = x_train.astype('float32')/255.0
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # flattens into vector of vectors of size 784
	x_test = x_test.astype('float32')/255.0
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # flattens into vector of vectors of size 784

	autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

	autoencoder.save('autoencoder_mnist.hdf5')

	visualise(autoencoder, x_test)
	


def visualise(autoencoder, x_test):
	'''
	Autoencoder is the model and x_test is the input
	'''

	# visualise the results
	random.shuffle(x_test)
	decoded_imgs = autoencoder.predict(x_test)

	n = 10  # how many digits we will display
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(x_test[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

if __name__ == '__main__':
	main()

