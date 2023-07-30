import numpy as np

class NBMM:
    """
    Negative Binomial Mixture Model
    """

    def __init__(self, r=2, n_components=2, max_iter=10):
        self.n_components = n_components
        self.r = r  # paper said that this can be fixed to 2
        self.pi = np.ones(n_components) / n_components
        self.max_iter = max_iter

    def fit(self, X: np.ndarray):
        N, G = X.shape # N: number of cells, G: number of genes
        self.p = self.r / (self.r + np.random.rand(G, self.n_components))

        # EM-algorithm
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

        return self.p

    def _e_step(self, X: np.ndarray):
        """
        Expectation step of the EM algorithm
        """
        ll = self._log_likelihood(X)
        maxima = np.argmax(ll, axis=1)
        z = np.eye(self.n_components)[maxima]
        return z

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        Maximization step of the EM algorithm
        """
        N_k = responsibilities.sum(axis=0)
        self.pi = N_k / N_k.sum()
        mu = (X.T @ responsibilities + 1e-4) / (N_k + 1)
        self.p = mu / (self.r + mu)

    def _log_likelihood(self, X: np.ndarray):
        """
        Compute the log likelihood of the data given the model current parameters
        """
        log_likelihood = 0
        log_likelihood = (
            np.log(self.pi)
            + X @ np.log(self.p)
            + np.sum(self.r * np.log(1 - self.p), axis=0)
        )
        return log_likelihood


if __name__ == "__main__":
    ### generate toy data for a NBMM

    # Set the ground truth parameters for each cluster
    true_p_1 = np.array(
        [0.7, 0.5, 0.3]
    )  # Success probabilities for cluster 1 (genes A, B, C)
    true_p_2 = np.array(
        [0.3, 0.6, 0.8]
    )  # Success probabilities for cluster 2 (genes A, B, C)
    true_r_1 = 2.0  # Dispersion parameter for cluster 1
    true_r_2 = 2.0  # Dispersion parameter for cluster 2

    # Set the number of data points in each cluster
    num_data_points_cluster1 = 100
    num_data_points_cluster2 = 100

    # Generate synthetic data for cluster 1
    data_cluster1 = np.zeros((num_data_points_cluster1, 3), dtype=int)
    data_cluster1[:, 0] = np.random.negative_binomial(
        n=true_r_1, p=true_p_1[0], size=num_data_points_cluster1
    )
    data_cluster1[:, 1] = np.random.negative_binomial(
        n=true_r_1, p=true_p_1[1], size=num_data_points_cluster1
    )
    data_cluster1[:, 2] = np.random.negative_binomial(
        n=true_r_1, p=true_p_1[2], size=num_data_points_cluster1
    )

    # Generate synthetic data for cluster 2
    data_cluster2 = np.zeros((num_data_points_cluster2, 3), dtype=int)
    data_cluster2[:, 0] = np.random.negative_binomial(
        n=true_r_2, p=true_p_2[0], size=num_data_points_cluster2
    )
    data_cluster2[:, 1] = np.random.negative_binomial(
        n=true_r_2, p=true_p_2[1], size=num_data_points_cluster2
    )
    data_cluster2[:, 2] = np.random.negative_binomial(
        n=true_r_2, p=true_p_2[2], size=num_data_points_cluster2
    )

    # Combine the data from both clusters
    data = np.vstack((data_cluster1, data_cluster2))

    # Create corresponding labels for the clusters (ground truth)
    labels = np.hstack(
        (
            np.zeros(num_data_points_cluster1, dtype=int),
            np.ones(num_data_points_cluster2, dtype=int),
        )
    )

    # Shuffle the data and labels to randomize the order
    shuffle_idx = np.random.permutation(data.shape[0])
    data = data[shuffle_idx]
    labels = labels[shuffle_idx]

    # Print the ground truth parameters for each cluster
    print("Cluster 1 - Success Probabilities:", true_p_1)
    print("Cluster 1 - Dispersion Parameter:", true_r_1)
    print("Cluster 2 - Success Probabilities:", true_p_2)
    print("Cluster 2 - Dispersion Parameter:", true_r_2)

    nbmm = NBMM(max_iter=100000)
    nbmm.fit(data)
    print(nbmm.p)
