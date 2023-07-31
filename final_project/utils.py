import matplotlib.pyplot as plt
import numpy as np


def choose_n_clusters(
    X: np.ndarray, mm_class, max_clusters: int = 10, min_n_clusters: int = 1
):
    """
    Chooses the number of clusters using BIC.
    returns: best_n_clusters and its corresponding predictions
    """
    min_bic = np.inf
    for n_clusters in range(min_n_clusters, max_clusters + 1):
        mm = mm_class(n_components=n_clusters)
        mm.fit(X)
        bic = mm.bic(X)
        if bic < min_bic:
            min_bic = bic
            best_n_clusters = n_clusters
            best_mm = mm
    print(f"Best result: {best_n_clusters} clusters, BIC = {min_bic}")
    return best_n_clusters, best_mm.predict(X)


def plot_2d_vis(results, title, clusters=None, transpose=False):
    """
    creates a 2d visualization of the results
    """
    n_norms = len(results.keys())
    n_trans = len(list(results.values())[0].keys())
    num_rows = n_norms if not transpose else n_trans
    num_cols = n_trans if not transpose else n_norms
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 5, num_rows * 5),
    )
    if axs.ndim == 1 and transpose:
        axs = axs.reshape(1, -1)
    elif axs.ndim == 1 and not transpose:
        axs = axs.reshape(-1, 1)

    fig.suptitle(title, fontsize=25)
    for i, (norm_key, sub_dict) in enumerate(results.items()):
        for j, (trans_key, result) in enumerate(sub_dict.items()):
            row, col = (i, j) if not transpose else (j, i)
            axs[row, col].scatter(
                result[:, 0],
                result[:, 1],
                s=7,
                c=clusters[norm_key][trans_key] if type(clusters) is dict else clusters,
                cmap="tab20" if type(clusters) is dict else None,
            )
            axs[row, col].set_title(f"{norm_key} {trans_key}".title())
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])

class NBMM:
    """
    Negative Binomial Mixture Model
    """

    def __init__(self, r=2, n_components=2, max_iter=1000):
        self.n_components = n_components
        self.r = r  # paper said that this can be fixed to 2
        self.pi = np.ones(n_components) / n_components
        self.max_iter = max_iter

        self.eps = 1e-4

    def fit(self, X: np.ndarray):
        N, G = X.shape
        # init with kmeans
        kmeans = KMeans(n_clusters=self.n_components, n_init=10).fit(X)
        mu = kmeans.cluster_centers_.T
        self.p = mu / (self.r + mu)
        # self.p = self.r / (self.r + np.random.rand(G, self.n_components))
        self.pi = np.bincount(kmeans.labels_) / N

        # EM-algorithm
        z = np.zeros((N, self.n_components))
        for _ in range(self.max_iter):
            old_z = z
            z = self._e_step(X)
            self._m_step(X, z)
            if (old_z == z).all():
                break

        return self.p

    def predict(self, X: np.ndarray):
        return self._e_step(X).argmax(axis=1)

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.predict(X)

    def _e_step(self, X: np.ndarray):
        """
        Expectation step of the EM algorithm
        """
        ll = self._log_likelihood(X)
        maxima = np.argmax(ll, axis=1)
        z = np.eye(self.n_components)[maxima]
        return z

    def _m_step(self, X: np.ndarray, z: np.ndarray):
        """
        Maximization step of the EM algorithm
        """
        N_k = z.sum(axis=0)
        self.pi = (N_k + self.eps) / N_k.sum()
        mu = (X.T @ z + 1e-4) / (N_k + 1)
        self.p = mu / (self.r + mu)

    def _log_likelihood(self, X: np.ndarray):
        """
        Compute the log likelihood of the data given the model current parameters
        """
        log_likelihood = 0
        log_likelihood = (
            np.log(self.pi + self.eps)
            + X @ np.log(self.p + self.eps)
            + np.sum(self.r * np.log(1 - self.p), axis=0)
        )
        return log_likelihood

    def bic(self, X: np.ndarray = None):
        """
        Compute the BIC of the data given the model current parameters
        """
        N, G = X.shape
        z = self._e_step(X)
        N_k = z.sum(axis=0)

        return np.sum(
            (G * np.log(N_k + self.eps)) / 2
            + np.sum(-self._log_likelihood(X) * z, axis=0)
        )


class NBMM:
    """
    Negative Binomial Mixture Model
    """

    def __init__(self, r=2, n_components=2, max_iter=1000):
        self.n_components = n_components
        self.r = r  # paper said that this can be fixed to 2
        self.pi = np.ones(n_components) / n_components
        self.max_iter = max_iter

        self.eps = 1e-4

    def fit(self, X: np.ndarray):
        N, G = X.shape
        # init with kmeans
        kmeans = KMeans(n_clusters=self.n_components, n_init=10).fit(X)
        mu = kmeans.cluster_centers_.T
        self.p = mu / (self.r + mu)
        # self.p = self.r / (self.r + np.random.rand(G, self.n_components))
        self.pi = np.bincount(kmeans.labels_) / N

        # EM-algorithm
        z = np.zeros((N, self.n_components))
        for _ in range(self.max_iter):
            old_z = z
            z = self._e_step(X)
            self._m_step(X, z)
            if (old_z == z).all():
                break

        return self.p

    def predict(self, X: np.ndarray):
        return self._e_step(X).argmax(axis=1)

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.predict(X)

    def _e_step(self, X: np.ndarray):
        """
        Expectation step of the EM algorithm
        """
        ll = self._log_likelihood(X)
        maxima = np.argmax(ll, axis=1)
        z = np.eye(self.n_components)[maxima]
        return z

    def _m_step(self, X: np.ndarray, z: np.ndarray):
        """
        Maximization step of the EM algorithm
        """
        N_k = z.sum(axis=0)
        self.pi = (N_k + self.eps) / N_k.sum()
        mu = (X.T @ z + 1e-4) / (N_k + 1)
        self.p = mu / (self.r + mu)

    def _log_likelihood(self, X: np.ndarray):
        """
        Compute the log likelihood of the data given the model current parameters
        """
        log_likelihood = 0
        log_likelihood = (
            np.log(self.pi + self.eps)
            + X @ np.log(self.p + self.eps)
            + np.sum(self.r * np.log(1 - self.p), axis=0)
        )
        return log_likelihood

    def bic(self, X: np.ndarray = None):
        """
        Compute the BIC of the data given the model current parameters
        """
        N, G = X.shape
        z = self._e_step(X)
        N_k = z.sum(axis=0)

        return np.sum(
            (G * np.log(N_k + self.eps)) / 2
            + np.sum(-self._log_likelihood(X) * z, axis=0)
        )