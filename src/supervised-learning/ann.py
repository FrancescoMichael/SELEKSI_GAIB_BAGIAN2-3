import numpy as np

class ArtificialNeuralNetwork:
    def __init__(self, layers, activations, batch_size=32, loss_function='binary_crossentropy', learning_rate=0.01, epochs=1000, regularization=None, optimizer='adam'):
        self.layers = layers
        self.activations = activations
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.optimizer = optimizer
        self._initialize_weights()
        if optimizer == 'adam':
            self._initialize_adam_parameters()

    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2 / self.layers[i])
            bias = np.zeros((1, self.layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def _initialize_adam_parameters(self):
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def _activation(self, x, func):
        if func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func == 'relu':
            return np.maximum(0, x)
        elif func == 'linear':
            return x
        elif func == 'softmax':
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)
        elif func == 'tanh':
            return np.tanh(x)
        elif func == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)

    def _activation_derivative(self, x, func):
        if func == 'sigmoid':
            return x * (1 - x)
        elif func == 'relu':
            return np.where(x > 0, 1, 0)
        elif func == 'linear':
            return np.ones_like(x)
        elif func == 'softmax':
            return np.exp(x) / (1 + np.exp(x))
        elif func == 'tanh':
            return 1 - x ** 2
        elif func == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)

    def _loss(self, y_true, y_pred):
        if self.loss_function == 'mse':
            loss = np.mean((y_true - y_pred) ** 2)
        elif self.loss_function == 'binary_crossentropy':
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        if self.regularization == 'l2':
            loss += 0.5 * np.sum([np.sum(w ** 2) for w in self.weights])
        elif self.regularization == 'l1':
            loss += np.sum([np.sum(np.abs(w)) for w in self.weights])
        return loss

    def _automatic_gradient(self, A, Z, y):
        gradients = []
        dA = A[-1] - y.reshape(-1, 1)
        for i in reversed(range(len(self.weights))):
            dZ = dA * self._activation_derivative(A[i+1], self.activations[i])
            dW = np.dot(A[i].T, dZ) / len(y)
            db = np.sum(dZ, axis=0, keepdims=True) / len(y)
            dA = np.dot(dZ, self.weights[i].T)
            if self.regularization == 'l2':
                dW += self.weights[i]
            elif self.regularization == 'l1':
                dW += np.sign(self.weights[i])
            gradients.insert(0, (dW, db))
        return gradients

    def _update_weights_adam(self, gradients):
        self.t += 1
        for i in range(len(self.weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients[i][0]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients[i][0] ** 2)
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients[i][1]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients[i][1] ** 2)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def fit(self, X, y):
        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                A = [X_batch]
                Z = []
                for i in range(len(self.weights)):
                    Z.append(np.dot(A[-1], self.weights[i]) + self.biases[i])
                    A.append(self._activation(Z[-1], self.activations[i]))

                gradients = self._automatic_gradient(A, Z, y_batch)

                if self.optimizer == 'adam':
                    self._update_weights_adam(gradients)
                else:
                    for i in range(len(self.weights)):
                        self.weights[i] -= self.learning_rate * gradients[i][0]
                        self.biases[i] -= self.learning_rate * gradients[i][1]

            y_pred = self.predict(X)
            loss = self._loss(y, y_pred)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        A = X
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self._activation(Z, self.activations[i])
        return A