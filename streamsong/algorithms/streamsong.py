import numpy as np
import matplotlib.pyplot as plt

class StreamSoNG():
    def __init__(self, ng, pknn, sp1m, typicality_threshold, learning_rate=0.1, neighborhood_range=2, num_points_for_new_class=10):
        self.ng = ng
        self.pknn = pknn
        self.sp1m = sp1m
        self.typicality_threshold = typicality_threshold
        self.learning_rate = learning_rate
        self.neighborhood_range = neighborhood_range
        self.num_points_for_new_class = num_points_for_new_class # M
        self.outliers = []
    
    def visualize(self, X, Y, show_radius=False):
        data_colors = ('lightskyblue', 'lightpink', 'mediumspringgreen', 'gold', 'mediumpurple')
        prototype_colors = ('blue', 'red', 'green', 'darkorange', 'darkviolet')

        fig, ax = plt.subplots(figsize=(10, 10))
        
        Y_pred = np.full_like(Y, np.nan)
        for i, (x, j) in enumerate(zip(X, Y)):
            Y_pred[i], typicality_class, _, s_typ, average_typicalities = self.pknn.predict(x, j)

        # data
        for c in range(self.pknn.num_classes):
            indices = np.where(Y_pred == c)
            X_c = X[indices]
            Y_c = Y_pred[indices]
            ax.scatter(X_c[:, 0], X_c[:, 1], label=f"data {c}", color=data_colors[c])

        # prototypes
        for c in range(self.pknn.num_classes):
            indices = np.where(self.pknn.prototype_labels == c)
            prototypes_c = self.pknn.prototypes[indices]
            ax.scatter(prototypes_c[:, 0], prototypes_c[:, 1], c=prototype_colors[c], label=f"prototypes {c}", marker="x")
            if show_radius:
                for i in range(len(prototypes_c)):
                    x = prototypes_c[i, 0]
                    y = prototypes_c[i, 1]
                    circle = plt.Circle((x, y), self.pknn.radius[c], color=prototype_colors[c], fill=False, linestyle="--")
                    ax.add_patch(circle)

        ax.set_aspect(1)
        plt.legend()
        plt.show()
    
    def update(self, k, prototype_idx, typicality_class, x):
        self.pknn.prototypes[prototype_idx] += self.learning_rate * typicality_class * np.exp((-k / self.neighborhood_range)) * (x - self.pknn.prototypes[prototype_idx])
    
    def remove_outliers(self):
        self.outliers.clear()
        
    def stream_process(self, X, Y):
        Y_pred = np.zeros(len(X))
        for t in range(len(X)):
            x = X[t]
            j = Y[t]
            
            class_closest, typicality_class, closest_indices, s_typicalities, _ = self.pknn.predict(x, j)
            Y_pred[t] = class_closest
            if typicality_class > self.typicality_threshold:
                for k, closest_idx in enumerate(closest_indices):
                    self.update(k, closest_idx, typicality_class, x)
            else:
                self.outliers.append(x)
                
                U, V = self.sp1m.process(self.outliers)

                new_cluster_typicalities = U[0]
                num_points_half_typicality = len(np.where(new_cluster_typicalities > 0.5)[0])
                if num_points_half_typicality > self.num_points_for_new_class:
                    print("> NEW CLASS from", num_points_half_typicality, "points")
                    X_c = self.outliers
                    Y_c = np.zeros(len(X_c)) + self.pknn.num_classes
                    self.ng.initialize_prototypes(X_c, Y_c)
                    for epoch in range(20):
                        self.ng.learn(X_c)
                    self.pknn.add_prototypes(self.ng.prototypes)
                    self.pknn.add_prototype_labels(self.ng.prototype_labels)
                    self.pknn.add_class()
                    self.pknn.estimate_radius(n=3)
                    self.pknn.radius
                    self.remove_outliers()
