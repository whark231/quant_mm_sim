import numpy as np
from scipy.stats import multivariate_normal

class HMM():
    def __init__(self, n_states=3, n_feats=3):
        self.n_states = n_states
        self.n_feats = n_feats
        self.pi = np.ones(n_states) / n_states # initial state distribution (n_states x 1)
        self.A = np.ones((n_states, n_states)) / n_states # transition matrix (n_states x n_states)
        self.mu = np.random.randn(n_states, n_feats) # emission means (n_states x n_feats)
        self.sigma = np.array([np.eye(n_feats) for _ in range(n_states)]) # emission covariances (n_states x n_features x n_featrues)

    # gives probabilities of seeing observation at time t in each state k (to be used later for bayes rule)
    def emission_prob(self, o_t):
        b = []
        for k in range(self.n_states):
            b.append(multivariate_normal.pdf(o_t, mean=self.mu[k], cov=self.sigma[k]))

        return np.array(b)

    def forward(self, observations):
        # observations: shape (T, n_feats)
        # returns: alpha of shape (T, n_states)
        
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        alpha[0] = self.pi * self.emission_prob(observations[0])

        for t in range(1,T):
            b = self.emission_prob(observations[t])
            alpha[t] = b * (alpha[t-1] @ self.A)

        return alpha
    
