import sys
import sklearn
import matplotlib
import numpy as np

from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import sklearn

def kmeans(data, n_clusters):
	estimator = MiniBatchKMeans(n_clusters = n_clusters)
	estimator.fit(data)
	
	# determine predicted labels
	# cluster_labels = infer_cluster_labels(estimator, )

	# print cluster metrics
	calculate_metrics(estimator, data)




	#
	# visualise centroids
	#

	# visualise_centroids(estimator.cluster_centers_)



	# reshape centroids into images
	rows = 10
	cols = 1
	centroids = estimator.cluster_centers_
	return centroids
	images = centroids.reshape(n_clusters, rows, cols)
	images = (images*255).astype(np.uint8)

	# create figure with subplots
	fig, axs = plt.subplots(cols, rows, figsize=(15, 10))

	# plt.gray()

	# loop through subplots and add centroid images
	for i, ax in enumerate(axs.flat):
		# determine infered label using cluster_labels dictionary
		# for key, value in cluster_labels.items():
		# 	if i in value:
		# 		ax.set_title('Inferred Labels: {}'.format(key))

		# add image to subplot
		ind = np.argpartition(estimator.cluster_centers_[i], -3)[-3:]
		print(ind)
		ax.matshow(images[i])
		ax.axis('off')

	plt.show()

def visualise_centroids(centroids):
	# first find distance between groups
	distances = np.array([]).reshape([len(centroids), len(centroids)-1])
	print(distances.shape())
	return
	for i in range(len(centroids)):
		for j in range(len(centroids)):
			centroidA = centroids[i]
			centroidB = centroids[j]




def calculate_metrics(estimator, data):
    # Calculate and print metrics
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    # print('Homogeneity: {}'.format(sklearn.metrics.homogeneity_score(labels, estimator.labels_)))

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