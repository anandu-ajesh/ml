import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
np.random.seed(0)
X=np.random.rand(25,2)
k=3
kmeans = KMeans(n_clusters=k,n_init=7)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(centers[:,0],centers[:,1], c='red', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()
