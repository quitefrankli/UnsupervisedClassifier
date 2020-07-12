import numpy as np
import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# some notes about tsne
# unlike PCA it uses a iterative and probabilistic method instead of 'hard-maths' like using eigen vectors
# it has O(n^2) complexity so it's recommended to have the number of dimensions be < 50, PCA can be used to reduce the initial dimensionality

digits = load_digits()
tsne = TSNE(n_components=3, verbose=1, n_iter=300)
tsne_results = tsne.fit_transform(digits.data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=digits.target, alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
plt.axis('off')
fig.colorbar(p)

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)


