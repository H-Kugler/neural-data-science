import numpy as np

class NBMixture():

    def __init__(self, n_components: int = 1, n_S: int = 10, r=2):
        self.n_components = n_components
        self.n_S = n_S
        self.S = np.arange(n_S)
        self.pi = np.ones(n_components) / n_components
        self.r = r

    def fit(self, X: np.ndarray):
        self.X = X
        self.n_cells, self.n_genes = self.X.shape
        # TODO think about init of these
        self.p = np.random.uniform(10, size=(self.n_genes, self.n_components))
        self.p /= self.p.sum(axis=1, keepdims=True)


        # EM-algorithm
        for _ in range(10):
            nbm._e_step()
            nbm._m_step()

    def predict(self, X: np.ndarray):
        pass

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.predict(self, X)
    
    def _e_step(self):
        log_lh = self._log_likelihood()
        print(log_lh)
        self.z = np.argmax(log_lh, axis=1)
        print(self.z)    
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
        print(self.pi, self.r, short_p, short_X)
        return np.log(self.pi) + (short_X @ np.log(short_p) + np.sum(self.r * np.log(1-short_p), axis=0))




if __name__ == "__main__":
    n_components = 5
    nbm = NBMixture(n_components=n_components, n_S=5)

    cells = np.array(['1', '2', '3', '4', '5', '6', '7', '8'])
    S = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    np.random.seed(246)
    X = np.random.randint(0, 2, size=(cells.shape[0], S.shape[0]))
    nbm.fit(X)