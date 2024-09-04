import numpy as np
from collections import deque

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def minkowski_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

class DBSCANScratch:
    def __init__(self, epsilon, min_samples, metric='euclidean', p=2):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.metric = metric
        self.p = p
        self.labels_ = None

    def _distance(self, x, y):
        if self.metric == 'euclidean':
            return euclidean_distance(x, y)
        elif self.metric == 'manhattan':
            return manhattan_distance(x, y)
        elif self.metric == 'minkowski':
            return minkowski_distance(x, y, self.p)
        else:
            raise ValueError("Unsupported metric")

    def _region_query(self, X, point_idx):
        neighbors = []
        for idx, point in enumerate(X):
            if self._distance(X[point_idx], point) <= self.epsilon:
                neighbors.append(idx)
        return neighbors

    def fit(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)

        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            visited[point_idx] = True

            neighbors = self._region_query(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id, visited)
                cluster_id += 1

        self.labels_ = labels

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id, visited):
        queue = deque(neighbors)
        labels[point_idx] = cluster_id

        while queue:
            current_point_idx = queue.popleft()

            if not visited[current_point_idx]:
                visited[current_point_idx] = True

                current_neighbors = self._region_query(X, current_point_idx)

                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)

            if labels[current_point_idx] == -1:
                labels[current_point_idx] = cluster_id