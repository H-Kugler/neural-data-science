import numpy as np

class NBMixture():

    def __init__(self, n_components: int = 1, n_S: int = None, maxiter: int = 20, r: float = 2):
        self.n_components = n_components
        self.n_S = n_S
        self.maxiter = maxiter
        self.r = r
        self.S = np.arange(n_S)
        self.pi = np.ones(n_components) / n_components


    def fit(self, X: np.ndarray):
        self.X = X
        self.n_cells, self.n_genes = self.X.shape
        
        self._run_em()

    def _run_em(self):
        
        self.n_cells, self.n_genes = self.X.shape

        # self.p = np.random.uniform(10, size=(self.n_genes, self.n_components))
        # self.p /= self.p.sum(axis=1, keepdims=True)

        self.z = np.arange(self.n_cells, dtype=int)

        # EM-algorithm
        self._m_step()
        for _ in range(self.maxiter):
            self._e_step()
            self._m_step()

    def predict(self, X: np.ndarray):
        self.X = X
        self._e_step()
        return self.z

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.predict(X)
    
    def _e_step(self):
        log_lh = self._log_likelihood()
        self.z = np.argmax(log_lh, axis=1)
        # this would instead yield one-hot encoding
        # z = np.zeros_like(log_lh)
        # z[np.arange(z.shape[0]), np.argmax(log_lh, axis=1)] = 1
        
    def _m_step(self):
        mean_p = self.p.mean(axis=1, keepdims=True)
        assigned_p = self.p[:, self.z]
        assert assigned_p.shape == (self.n_genes, self.n_cells)
        
        # TODO check this formula
        Y = [X[:, g] @ (np.log(assigned_p) - np.log(mean_p))[g] for g in range(self.n_genes)]
        Y += np.sum(self.r * (np.log(1 - assigned_p) - np.log(1 - mean_p)), axis=1)

        assert Y.shape == (self.n_genes, )
        self.S = np.argsort(Y)[::-1][:self.n_S]
        
        A = 1e-4
        B = 1
        mu = np.zeros_like(self.p)
        N_k = np.zeros(self.n_components)
        for k in range(self.n_components):
            N_k[k] = np.sum(self.z == k)
            mu[:, k] = (self.X[self.z == k, :].sum(axis=0) + A) / B + N_k[k]
        self.p = mu / (self.r + mu)

        self.pi = N_k / self.n_cells

    
    def _log_likelihood(self):
        short_p = self.p[self.S, :]
        short_X = self.X[:, self.S]
        return np.log(self.pi) + (short_X @ np.log(short_p) + np.sum(self.r * np.log(1-short_p), axis=0))
    
    def score(self, X: np.ndarray):
        self.X = X
        return self._log_likelihood().mean()


if __name__ == "__main__":
    n_components = 5
    nbm = NBMixture(n_components=n_components, n_S=5)

    cells = np.array(['1', '2', '3', '4', '5', '6', '7', '8'])
    genes = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    np.random.seed(246)
    X = np.random.randint(0, 2, size=(cells.shape[0], genes.shape[0]))
    print(nbm.fit_predict(X))
    print(nbm.score(X))