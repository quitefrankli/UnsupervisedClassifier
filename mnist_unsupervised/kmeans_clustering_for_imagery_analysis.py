import sys
import sklearn
import matplotlib
import numpy as np

from matplotlib import pyplot as plt
from keras.datasets import mnist
from sklearn.cluster import MiniBatchKMeans
import sklearn

def main():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# preprocessing the images
	# convert each image to 1 dimensional array
	X = x_train.reshape(len(x_train),-1)
	Y = y_train
	# normalize the data to 0 - 1
	X = X.astype(float) / 255.

	n_digits = len(np.unique(y_test))

	clusters = [10, 16, 36, 64, 144]

	# test different numbers of clusters
	for n_clusters in clusters:
		estimator = MiniBatchKMeans(n_clusters = n_clusters)
		estimator.fit(X)
		
		# determine predicted labels
		cluster_labels = infer_cluster_labels(estimator, Y)
		predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)

		# print cluster metrics
		calculate_metrics(estimator, X, Y)
		# calculate and print accuracy
		print('Accuracy: {}\n'.format(sklearn.metrics.accuracy_score(Y, predicted_Y)))

		#
		# visualise centroids
		#

		centroids = estimator.cluster_centers_
		
		# reshape centroids into images
		images = centroids.reshape(n_clusters, 28, 28)
		images = (images*255).astype(np.uint8)

		# create figure with subplots
		l = np.sqrt(n_clusters)
		if not l.is_integer():
			continue
		fig, axs = plt.subplots(int(l), int(l), figsize=(20, 20))

		plt.gray()

		# loop through subplots and add centroid images
		for i, ax in enumerate(axs.flat):
			# determine infered label using cluster_labels dictionary
			for key, value in cluster_labels.items():
				if i in value:
					ax.set_title('Inferred Labels: {}'.format(key))

			# add image to subplot
			ax.matshow(images[i])
			ax.axis('off')

		plt.show()

def calculate_metrics(estimator, data, labels):
    # Calculate and print metrics
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(sklearn.metrics.homogeneity_score(labels, estimator.labels_)))

def infer_cluster_labels(kmeans, actual_labels):
	"""
	Associates most probable label with each cluster in KMeans model
	returns: dictionary of clusters assigned to each label
	"""

	inferred_labels = {}

	for i in range(kmeans.n_clusters):

		# find index of points in cluster
		labels = []
		index = np.where(kmeans.labels_ == i)

		# append actual labels for each point in cluster
		labels.append(actual_labels[index])

		# determine most common label
		if len(labels[0]) == 1:
			counts = np.bincount(labels[0])
		else:
			counts = np.bincount(np.squeeze(labels))

		# assign the cluster to a value in the inferred_labels dictionary
		if np.argmax(counts) in inferred_labels:
			# append the new number to the existing array at this slot
			inferred_labels[np.argmax(counts)].append(i)
		else:
			# create a new array in this slot
			inferred_labels[np.argmax(counts)] = [i]
		
	return inferred_labels  

def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

if __name__ == "__main__":
	main()