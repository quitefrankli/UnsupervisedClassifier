import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disables extranneous logging

from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras.datasets import mnist

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import numpy as np
import random

from kmeans import kmeans

def main():
	# ----------------
	# Input data

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# normalise input data
	x_train = x_train.astype('float32')/255.0
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # flattens into vector of vectors of size 784
	x_test = x_test.astype('float32')/255.0
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # flattens into vector of vectors of size 784

	model_file_name = 'autoencoder_mnist.hdf5'
	try:
		autoencoder = load_model(model_file_name)
	except OSError: # perhaps file does not exist
		print('File not found, creating new file')
		autoencoder = generate_model(x_train, x_test)
		autoencoder.save(model_file_name)

	# ----------------
	# to get the output of the hidden layer ie. the encoded layer we duplicate the same model but truncated
	encoder = Sequential()
	encoder.add(autoencoder.layers[0])
	
	# random_seed = random.randint(0, 1e6)
	# random.Random(random_seed).shuffle(x_test)
	# random.Random(random_seed).shuffle(y_test)

	# use kmeans clustering find real value
	encoded = encoder.predict(x_train)
	print(encoded.shape)
	centroids = kmeans(data=encoded, n_clusters=10)

	# now visualise the clusters using tsne
	# tsne_model = TSNE(n_components=2, verbose=1, n_iter=300)
	# n = 5000
	# x = x_train[:n]
	# y = y_train[:n]
	# tsne_results = tsne_model.fit_transform(x)
	# plt.scatter(
	# 	tsne_results[:, 0], tsne_results[:, 1],
	# 	alpha=0.5,
	# )

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# p = ax.scatter(
	# 	tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], 
	# 	alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10),
	# 	c=y
	# )
	# plt.axis('off')
	# fig.colorbar(p)
	plt.show()

	return

	# n = 10
	# inputs = x_test[:n]
	# predictions = get_prediction(encoder, inputs)
	# print('predictions are', predictions, 'labels are', y_test[:n])



	# ----------------
	# Visualisation

	visualise(autoencoder, x_test)

def get_prediction(encoder, inputs):
	'''
	Returns list of predicted indices
	These indices do not mean anything and must be interpretted to be useful
	'''

	predictions = encoder.predict(inputs)
	print(predictions)
	predictions = np.argmax(predictions, axis=1)

	return predictions


def generate_model(x_train, x_test):
	# ----------------
	# Model

	# size of encoded representation, aka hidden layer
	encoding_dim = 10

	autoencoder = Sequential()
	autoencoder.add(Dense(encoding_dim, input_shape=(784,), activation='relu'))
	autoencoder.add(Dense(784, activation='sigmoid'))

	# train the autoencoder
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

	return autoencoder

def visualise(autoencoder, x_test):
	'''
	Autoencoder is the model and x_test is the input
	'''

	# visualise the results
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

