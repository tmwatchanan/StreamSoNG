import numpy as np
import matplotlib.pyplot as plt

class NeuralGas():
    def __init__(self, num_prototypes, dimension, step_size, neighborhood_range):
        self.num_prototypes = num_prototypes
        self.dimension = dimension
#         self.prototypes = np.random.random((num_prototypes, dimension))
        self.prototypes = np.random.normal(0, 4, (num_prototypes, dimension))
        self.prototype_labels = np.full(num_prototypes, np.nan)
        self.step_size = step_size
        self.neighborhood_range = neighborhood_range
    
    def initialize_prototypes(self, X, Y):
        indices = np.random.choice(len(X), self.num_prototypes)
        for i in range(len(indices)):
            self.prototypes[i] = X[indices[i]]
            self.prototype_labels[i] = Y[indices[i]]
    
    def visualize(self, X, Y):
        fig, ax = plt.subplots()
        
        # data
        ax.scatter(X[:, 0], X[:, 1], label="data", c=Y, cmap="Pastel1")
        
        # prototypes
        ax.scatter(self.prototypes[:, 0], self.prototypes[:, 1], c=self.prototype_labels, cmap="Set1", label="prototypes", marker="x")
        
        plt.show()
        
    def get_distance(self, a, b):
        return np.sum(np.power(a - b, 2))
    
    def learn(self, X):
        idx = np.random.choice(len(X))
        x = X[idx]
        
        distances = np.zeros(self.num_prototypes)
        for i in range(len(distances)):
            distances[i] = self.get_distance(x, self.prototypes[i])
        indices = np.argsort(distances)
        
        for k, i in enumerate(indices, 1):
            delta = self.step_size * np.exp(-k / self.neighborhood_range) * (x - self.prototypes[i])
            self.prototypes[i] += delta

if __name__ == '__main__':
    X1 = np.random.normal(-2, 2,(100, 2))
    Y1 = np.zeros(len(X1))
    X2 = np.random.normal(15, 4,(100, 2))
    Y2 = np.zeros(len(X2)) + 1
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    num_prototypes = 5
    dimension = 2
    prototypes = []
    prototype_labels = []

    for c in range(C):
        indices = np.where(Y == c)
        X_c = X[indices]
        Y_c = Y[indices]
        ng = NeuralGas(num_prototypes=num_prototypes, dimension=dimension, step_size=0.01, neighborhood_range=2)
        ng.initialize_prototypes(X_c, Y_c)
        for epoch in range(20):
            ng.learn(X)
        prototypes.extend(ng.prototypes)
        prototype_labels.extend(ng.prototype_labels)

    prototypes = np.array(prototypes)
    prototype_labels = np.array(prototype_labels)
    print(prototypes.shape, prototype_labels.shape)
