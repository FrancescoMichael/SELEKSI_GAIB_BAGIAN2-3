import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifierScratch:
    def __init__(self, n_trees=100, max_depth=None, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(X.shape[1]))
            else:
                max_features = X.shape[1]

            features = np.random.choice(range(X.shape[1]), max_features, replace=False)
            tree.fit(X_sample[:, features], y_sample)
            self.trees.append((tree, features))

    def predict_one(self, x):
        tree_preds = np.array([tree.predict(x[features].reshape(1, -1))[0] for tree, features in self.trees])
        return np.bincount(tree_preds).argmax()

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])