import numpy as np
import random

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import load_digits

digits = load_digits()
pca = PCA(n_components=3)
projected = pca.fit_transform(digits.data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c=digits.target, alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
plt.axis('off')

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)