from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import random

def main():
	# size of encoded representation
	encoding_dim = 32

	# input placeholder
	input_img = Input(shape=(784,))

	# encoded representation of the input
	layer = Dense(encoding_dim, activation='relu')
	encoded = layer(input_img)

	# decoded reconstruction of the input
	layer = Dense(784, activation='sigmoid')
	decoded = layer(encoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# -----------------

	# encoder model
	encoder = Model(input_img, encoded)

	# -----------------

	# decoder model
	encoded_input = Input(shape=(encoding_dim,))
	decoder_layer = autoencoder.layers[-1] # last layer of autoencoder
	decoder = Model(encoded_input, decoder_layer(encoded_input))

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

	# -----------------
	
	# visualise the results
	random.shuffle(x_test)
	encoded_imgs = encoder.predict(x_test)
	decoded_imgs = decoder.predict(encoded_imgs)

	n = 30  # how many digits we will display
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

