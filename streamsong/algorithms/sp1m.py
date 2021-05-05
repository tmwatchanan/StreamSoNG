import numpy as np

class SP1M():
    def __init__(self, C=1, fuzzifier=2, bandwidth=np.sqrt(2)):
        self.C = C
        self.fuzzifier = fuzzifier # m
        # TODO: dynamic η for each cluster
        self.bandwidth = bandwidth # η
        
        assert self.fuzzifier >= 1
        assert self.bandwidth > 0

    def pick_cluster_center(self, c, X, partition_matrix):
        n = len(X)
        if c == 0:
            idx = np.random.choice(n)
        else:
            # probabilities based on typicalities
            probabilities = np.zeros(n)
            # TODO:
            partition_matrix = np.array(partition_matrix)
            if partition_matrix.ndim == 1:
                partition_matrix = partition_matrix[:, None]
            max_u = np.amax(partition_matrix, axis=0)
            probabilities = (1 - max_u) / (n - np.sum(max_u))
            idx = np.random.choice(n, p=probabilities)
        return X[idx]
    
    def compute_distance(self, a, b, squared=True):
        d = np.sum((a - b) ** 2)
        if squared:
            return d
        else:
            return np.sqrt(d)
        
    def compute_typicality(self, v, x):
        distance = self.compute_distance(v, x, squared=True)
        denom = 1 + ((distance / self.bandwidth) ** (1 / (self.fuzzifier - 1)))
        return 1 / denom
    
    def compute_cluster_center(self, cluster_typicalities, X):
        n = len(X)
        cluster_typicalities **= self.fuzzifier
        if len(cluster_typicalities) != len(X):
            raise ValueError("# of elements in u_ik and x_k are not the same")
        v = np.sum(cluster_typicalities[:, None] * X, axis=0) / np.sum(cluster_typicalities)
        return v
    
    def process(self, X):
        """run SP1M on the data X
        
        :param X: numerical feature data set
        :type X: numpy array
        """
        n = len(X)
        partition_matrix = [] # U
        cluster_centers = [] # V
        
        # loop until C cluster centers were found
        for c in range(self.C):
            # first step, each data point has the same probability to be chosen
            # as the initial cluster center v
            v = self.pick_cluster_center(c, X, partition_matrix)
            
            typicalities = np.zeros(n)
        
            while True:
                for k in range(n):
                    typicalities[k] = self.compute_typicality(v, X[k])
                v = self.compute_cluster_center(typicalities, X)
                
                # check convergence
                prototype_distances = np.full(len(cluster_centers), np.nan)
                for i, center in enumerate(cluster_centers):
                    prototype_distances[i] = self.compute_distance(v, center, squared=False)
                if len(prototype_distances) == 0 or min(prototype_distances) >= 2 * self.bandwidth:
                    break
                
            partition_matrix.append(typicalities)
            cluster_centers.append(v)
            
        assert len(partition_matrix) == len(cluster_centers)
        return partition_matrix, cluster_centers

if __name__ == '__main__':
    X1 = np.random.normal(-2, 2,(100, 2))
    Y1 = np.zeros(len(X1))
    X2 = np.random.normal(15, 4,(100, 2))
    Y2 = np.zeros(len(X2)) + 1
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    sp1m = SP1M(C=2, fuzzifier=2, bandwidth=np.sqrt(2))
    U, V = sp1m.process(X)
    print("V", V)
    print("U", np.array(U).shape)
