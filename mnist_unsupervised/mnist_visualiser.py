# this script visualises MNIST dataset using TSNE

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disables extranneous logging

from keras.datasets import mnist

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import numpy as np
import random

def main():
	# ----------------
	# Input data

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# normalise input data
	x_train = x_train.astype('float32')/255.0
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # flattens into vector of vectors of size 784
	x_test = x_test.astype('float32')/255.0
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # flattens into vector of vectors of size 784

	n = 1000
	x = x_train[:n]
	y = y_train[:n]
	tsne_visualiser = TSNE_Visualiser(x, y)
	tsne_visualiser.visualise()

class TSNE_Visualiser():
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.fig = plt.figure(figsize=(12, 12))
		self.n_expected_clusters = 10

	def visualise(self):
		self.visualise_2d()
		self.visualise_3d()
		plt.show()

	def visualise_2d(self):
		tsne = TSNE(n_components=2, n_iter=300)
		tsne_results = tsne.fit_transform(self.x)

		# no color
		ax = self.fig.add_subplot(2, 2, 1)
		ax.scatter(
			tsne_results[:, 0], tsne_results[:, 1],
			alpha=0.5
		)
		plt.axis('off')

		# color
		ax = self.fig.add_subplot(2, 2, 2)
		ax.scatter(
			tsne_results[:, 0], tsne_results[:, 1],
			alpha=0.5, cmap=plt.cm.get_cmap('Spectral', self.n_expected_clusters),
			c=self.y
		)
		plt.axis('off')

	def visualise_3d(self):
		tsne = TSNE(n_components=3, n_iter=300)
		tsne_results = tsne.fit_transform(self.x)

		# no color
		ax = self.fig.add_subplot(2, 2, 3, projection='3d')
		ax.scatter(
			tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], 
			alpha=0.5
		)
		plt.axis('off')

		# color
		ax = self.fig.add_subplot(2, 2, 4, projection='3d')
		p = ax.scatter(
			tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], 
			alpha=0.5, cmap=plt.cm.get_cmap('Spectral', self.n_expected_clusters),
			c=self.y
		)
		plt.axis('off')
		self.fig.colorbar(p)




if __name__ == '__main__':
	main()

