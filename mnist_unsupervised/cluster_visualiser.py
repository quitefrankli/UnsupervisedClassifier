import numpy as np
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def main():
	desired_cluster_locations = np.array([
		[1, 1, 1],
		[3, 2, 1],
		[6, 6, 5]
	])
	data = generate_data(desired_cluster_locations=desired_cluster_locations, std_dev=0.3, n=10)
	data=data.reshape(int(len(data)/3), 3)

	# the original similiarity matrix, we want the reduced dimensionality result to be as close to this as possible
	# original_similarity_matrix = get_similarity_matrix(data)

	estimator = MiniBatchKMeans(n_clusters=3)
	estimator.fit(data)

	# keep in mind that these centroids may have a very large dimensionality
	centroids = estimator.cluster_centers_

	# original_centroid_simularity_matrix = get_similarity_matrix(centroids)

	X_embedded = TSNE(n_components=2).fit_transform(data)
	x = X_embedded

	y = np.zeros(shape=x.shape)
	plt.scatter(x=x[:, 0], y=x[:, 1])
	plt.show()

	return
	visualise_centroids_1D(centroids, data)

def visualise_centroids_1D(centroids, original_simularity_matrix, raw_data):
	# let first centroid be at origin and the following centroids be d-distance away from it on a straight line
	origin = centroids[0]

	for centroid in centroids:
		distance = np.linalg.norm(centroid - origin)
		plt.scatter(distance, 0)

	# now for reference we would also like to see the raw data points on the same plane


	plt.show()

def get_similarity_matrix(data):
	n_data = len(data)
	matrix = np.zeros(shape=(n_data, n_data))

	for row, x in enumerate(data):
		for col, y in enumerate(data):
			matrix[row][col] = np.linalg.norm(x-y)

	return matrix

def get_difference_between_matrices(A, B):
	absolute_difference = np.absolute(A - B)
	return absolute_difference.sum()

def generate_data(desired_cluster_locations, std_dev, n):
	'''
	desired_cluster_locations = numpy array
	variance = expected variance of points surrounding the cluster
	n = number of points per cluster
	'''

	data = np.array([])
	# the dimensionality can be inferred from the dcls
	for dcl in desired_cluster_locations:
		for _ in range(n):
			x = np.random.normal(loc=dcl, scale=std_dev)
			data = np.append(data, x)

	return data

if __name__ == "__main__":
	main()