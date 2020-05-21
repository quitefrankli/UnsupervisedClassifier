from ClusteringLayer import ClusteringLayer

import keras

model = keras.Sequential()
model.add(ClusteringLayer(n_clusters=10))


