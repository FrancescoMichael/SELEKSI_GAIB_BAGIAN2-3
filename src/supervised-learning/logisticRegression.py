import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=100, regularization=None, loss_function='log_loss', method='gradient_descent'):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.loss_function = loss_function
        self.method = method
    
    def init_params(self, n):
        self.bias = 0
        self.weights = np.zeros(n)
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_target, y_pred, X):
        if self.loss_function == 'hinge_loss':
            y_target = np.where(y_target <= 0, -1, 1)
            margins = 1 - y_target * (np.dot(X, self.weights) + self.bias)
            hinge_loss = np.maximum(0, margins)
            return np.mean(hinge_loss)
        elif self.loss_function == 'log_loss':
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_target * np.log(y_pred) + (1 - y_target) * np.log(1 - y_pred))
        
    def compute_gradient(self, X, y, y_pred):
        m = len(y)
        dw = np.dot(X.T, (y_pred - y)) / m
        db = np.sum(y_pred - y) / m
        if self.regularization == 'l1':
            dw += (1/m) * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += ((1/m) * self.weights)
        return dw, db

    def compute_hessian(self, X, y_pred):
        m, n = X.shape
        S = np.diag(y_pred * (1 - y_pred))
        H = np.dot(np.dot(X.T, S), X) / m
        if self.regularization == 'l2':
            H += np.identity(n) / m
        return H
    
    def fit(self, X_train, y_train):
        samples, features = X_train.shape
        self.init_params(features)

        for _ in range(self.iterations):
            z = np.dot(X_train, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            loss = self.compute_loss(y_train, y_pred, X_train)

            dw, db = self.compute_gradient(X_train, y_train, y_pred)
            if self.method == 'gradient_descent':
                self.weights -= (self.learning_rate * dw)
                self.bias -= (self.learning_rate * db)
            elif self.method == 'newton':
                H = self.compute_hessian(X_train, y_pred)
                H_inv = np.linalg.inv(H)
                update = np.dot(H_inv, dw)
                self.weights -= update
                self.bias -= db
    
    def predict(self, X_test):
        z = np.dot(X_test, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return np.where(y_pred >= 0.5, 1, 0)