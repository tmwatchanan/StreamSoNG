import numpy as np
import matplotlib.pyplot as plt

class PKNN():
    def __init__(self, prototypes, prototype_labels, num_classes, K, fuzzifier):
        self.prototypes = prototypes
        self.prototype_labels = prototype_labels.astype(int)
        self.num_classes = num_classes
        self.K = K
        self.fuzzifier = fuzzifier
    
    def add_prototypes(self, new_prototypes):
#         self.prototypes = np.append(self.prototypes, new_prototypes, axis=0)
        self.prototypes = np.concatenate((self.prototypes, new_prototypes))
        
    def add_prototype_labels(self, new_prototype_labels):
#         self.prototype_labels = np.append(self.prototype_labels, new_prototype_labels).astype(int)
        self.prototype_labels = np.concatenate((self.prototype_labels, new_prototype_labels)).astype(int)
    
    def add_class(self):
        self.num_classes += 1
    
    def num_prototypes(self):
        return self.prototypes.shape[0]
    
    def get_distance(self, a, b):
        return np.sum(np.power(a - b, 2))
    
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum(np.power(a - b, 2)))
    
    def knn(self, A, B, n=5, ignore_diag=False, distance_fx=None):
        if A.ndim == 1:
            A = A[None, :]
        if distance_fx is None:
            distance_fx = self.get_distance
        distances = np.full((len(A), len(B)), np.nan)
        for i in range(len(A)):
            for j in range(len(B)):
                if ignore_diag and i == j:
                    continue
                distances[i, j] = distance_fx(A[i], B[j])
            
        sorted_indices = np.argpartition(distances, kth=range(n), axis=1)[:, :n]
        shortest_distances= np.full((len(A), n), np.nan)
        for i in range(len(shortest_distances)):
            shortest_distances[i] = distances[i, sorted_indices[i]]
        return sorted_indices, shortest_distances
        
    def estimate_radius(self, n=5, show_figure=False):
        if n >= self.num_prototypes():
            raise ValueError(f"n({n}) must be less than the number of prototypes({self.num_prototypes()})")
        self.radius = np.full(self.num_classes, np.nan)
        for c in range(self.num_classes):
            indices_c = np.where(self.prototype_labels == c)
            prototypes_c = self.prototypes[indices_c]
            _, shortest_distances = self.knn(prototypes_c, prototypes_c, n=n, ignore_diag=True, distance_fx=self.euclidean_distance)
            
            hist, bins = np.histogram(shortest_distances, bins=8)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            self.radius[c] = mean = np.mean(center)
            
            if show_figure:
                fig, ax = plt.subplots()
                ax.bar(center, hist, align='center', width=width)
                ax.plot([mean] * (np.max(hist)+1), np.arange(np.max(hist)+1), color="red")
                ax.set_title(f"Class {c} $\eta = {mean:.3f}$")
    
    def compute_fuzzy_membership(self, n_i, pred, actual):
        if pred == actual:
            return 0.51 + (n_i / self.K) * 0.49
        else:
            return (n_i / self.K) * 0.49
    
    def compute_typicality(self, x, p, i):
        return 1 / (1 + (self.get_distance(x, p) / self.radius[i]) ** (1 / (self.fuzzifier-1)))
    
    def fuzzy_typicality(self, mu, t):
        return mu * t

    def average_typicality(self, fuzzy_typicalities):
        if np.sum(fuzzy_typicalities) == 0:
            return 0
        return np.nan_to_num(np.mean(fuzzy_typicalities))

    def s_function(self, x, a=0, b=0.5, c=1):
        if x <= 0:
            return 0
        elif a < x and x <= b:
            return ((x - a)**2) / (2 * ((b - a)**2))
        elif b < x and x <= c:
            return (-((x - c)**2) / (2 * ((b - c)**2))) + 1
        elif x > c:
            return 1
    
    def predict(self, x, j):
        sorted_indices, _ = self.knn(x, self.prototypes, n=self.K)
        t = 0

        # crisp membership ========================================================
        crisp_memberships = np.zeros(self.num_classes) # n_i
        for i in self.prototype_labels[sorted_indices[t]]:
            crisp_memberships[i] += 1
        pred_class = np.argmax(crisp_memberships)
        assert np.sum(crisp_memberships) == self.K

        # fuzzy membership ========================================================
        fuzzy_memberships = np.full(sorted_indices.shape[1], np.nan)
        for k, i in enumerate(self.prototype_labels[sorted_indices[t]]):
            fuzzy_memberships[k] = self.compute_fuzzy_membership(n_i=crisp_memberships[i], pred=i, actual=j)
        assert np.min(fuzzy_memberships) >= 0
        assert np.max(fuzzy_memberships) <= 1

        # typicality ==============================================================
        typicalities = np.full_like(fuzzy_memberships, np.nan)
        for k, idx in enumerate(sorted_indices[t]):
            p = self.prototypes[idx]
            i = self.prototype_labels[idx]
            typicalities[k] = self.compute_typicality(x, p, i)
        assert np.min(typicalities) >= 0
        assert np.max(typicalities) <= 1

        # fuzzy typicality ========================================================
        fuzzy_typicalities = np.full_like(typicalities, np.nan)
        for k, idx in enumerate(sorted_indices[t]):
            fuzzy_typicalities[k] = self.fuzzy_typicality(fuzzy_memberships[k], typicalities[k])
        assert np.min(fuzzy_typicalities) >= 0
        assert np.max(fuzzy_typicalities) <= 1

        # average typicality ======================================================
        pred_classes = self.prototype_labels[sorted_indices[t]]
        average_typicalities = np.full_like(crisp_memberships, np.nan)
        for i in range(self.num_classes):
            indices = np.where(pred_classes == i)
            average_typicalities[i] = self.average_typicality(fuzzy_typicalities[indices])
        assert np.min(average_typicalities) >= 0
        assert np.max(average_typicalities) <= 1

        # S function ==============================================================
        s_typicalities = np.full_like(average_typicalities, np.nan)
        for i in range(self.num_classes):
            s_typicalities[i] = self.s_function(average_typicalities[i])

        # class_typicality =========================================================
        class_closest = np.argmax(s_typicalities)
        typicality_class = s_typicalities[class_closest]
        return class_closest, typicality_class, sorted_indices[t], s_typicalities, average_typicalities
