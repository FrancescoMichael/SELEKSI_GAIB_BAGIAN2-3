import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters, init='random', max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init

    def _initialize_centroids(self, X):
        if self.init == 'random':
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            centroids = X[indices]
        elif self.init == 'k-means++':
            centroids = []
            centroids.append(X[np.random.randint(X.shape[0])])
            for _ in range(1, self.n_clusters):
                distances = np.min(np.array([np.linalg.norm(X - c, axis=1) for c in centroids]), axis=0)
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
                for i, p in enumerate(cumulative_probabilities):
                    if r < p:
                        centroids.append(X[i])
                        break
            centroids = np.array(centroids)
        else:
            raise ValueError("Unknown initialization method")
        return centroids

    def _assign_clusters(self, X, centroids):
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
        return np.argmin(distances, axis=0)

    def _compute_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def fit(self, X):
        centroids = self._initialize_centroids(X)
        for _ in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._compute_centroids(X, labels)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids
        self.labels = labels