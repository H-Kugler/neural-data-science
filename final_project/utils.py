import matplotlib.pyplot as plt
import numpy as np


def plot_2d_vis(results, title, clusters=None):
    # plot t-SNE
    num_rows = len(results.keys())
    num_cols = len(list(results.values())[0].keys())
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 5, num_rows * 5),
    )
    if axs.ndim == 1:
        axs = axs.reshape(-1, 1)

    fig.suptitle(title, fontsize=25)
    for i, (norm_key, sub_dict) in enumerate(results.items()):
        for j, (trans_key, result) in enumerate(sub_dict.items()):
            axs[i, j].scatter(
                result[:, 0],
                result[:, 1],
                s=7,
                c=clusters[norm_key][trans_key]
                if clusters is dict
                else clusters[norm_key][trans_key],
                cmap="tab20" if clusters is dict else None,
            )
            axs[i, j].set_title(f"{norm_key} {trans_key}".title())
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])



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
        self.p = self.r / (self.r + np.random.rand(G, self.n_components))

        # EM-algorithm
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

        return self.p

    def predict(self, X: np.ndarray):
        return self._e_step(X)

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

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        Maximization step of the EM algorithm
        """
        N_k = responsibilities.sum(axis=0)
        self.pi = (N_k + self.eps) / N_k.sum()
        mu = (X.T @ responsibilities + 1e-4) / (N_k + 1)
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
    
