import numpy as np

class SVMScratch:
    def __init__(self, kernel='rbf', C=1.0, learning_rate=0.001, n_iters=1000, degree=3, gamma=0.5, coef0=1):
        self.kernel = kernel
        self.C = C
        self.lr = learning_rate
        self.n_iters = n_iters
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.X = X
        self.y = y_

        self.K = self.compute_kernel(X, X)
        
        for _ in range(self.n_iters):
            for i in range(n_samples):
                decision_value = np.sum(self.alpha * y_ * self.K[:, i]) + self.b
                
                if y_[i] * decision_value < 1:
                    self.alpha[i] += self.lr * (1 - y_[i] * decision_value - self.C * self.alpha[i])
                    self.b += self.lr * y_[i]
                else:
                    self.alpha[i] -= self.lr * self.C * self.alpha[i]
                
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)
        
        support_vector_idx = np.where((self.alpha > 1e-5) & (self.alpha < self.C))[0]
        if len(support_vector_idx) > 0:
            self.b = np.mean([y_[i] - np.dot(self.alpha * y_, self.K[:, i]) for i in support_vector_idx])
        else:
            self.b = 0

    def compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            sq_dists = -2 * np.dot(X1, X2.T) + np.sum(X2 ** 2, axis=1) + np.sum(X1 ** 2, axis=1)[:, np.newaxis]
            return np.exp(-self.gamma * sq_dists)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
        else:
            raise ValueError("Unknown kernel type")
    
    def predict(self, X):
        K = self.compute_kernel(X, self.X)
        decision_values = np.dot(K, self.alpha * self.y) + self.b
        return np.where(decision_values >= 0, 1, 0)
