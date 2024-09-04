import numpy as np
from collections import Counter

class KNNScratch:
    def __init__(self, neighbors=5, metric='euclidean'):
        self.neighbors = neighbors
        self.metric = metric
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum(np.square(x1 - x2))) 
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2)) 
        elif self.metric == 'minkowski':
            p = 5
            return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)
        
    def predict(self, X_test):
        predictions = []

        for x_test in X_test:
            distances = [self.distance(x_test, x_train) for x_train in self.X_train]
            sorted_distances_indexes = np.argsort(distances)[:self.neighbors]
            k_nearest = [self.y_train[i] for i in sorted_distances_indexes]
            most_class = Counter(k_nearest).most_common(1)
            predictions.append(most_class[0][0])
        
        return predictions