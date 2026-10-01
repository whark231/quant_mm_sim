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

    # gives probabilities of seeing observation at time t given state k (to be used later for bayes rule)
    def emission_prob(self, o_t):
        b = []
        for k in range(self.n_states):
            b.append(multivariate_normal.pdf(o_t, mean=self.mu[k], cov=self.sigma[k]))

        return np.array(b)

    # Captures joint probability of being in state k and seeing observations up until time t
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
    
    # Captures conditional probability of each future observation up until time T given state k at time t
    def backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # base case
        beta[T-1] = 1

        # recursion
        for t in range(T-2, -1, -1):
            b = self.emission_prob(observations[t+1])
            beta[t] = self.A @ (b * beta[t+1])

        return beta

    # Result of bayes rule: probability of being in state k given all of the observations
    def compute_gamma(self, alpha, beta):
        # shape: (T, n_states)
        return (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)

    def compute_xi(self, observations, alpha, beta):
        T = len(observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))

        for t in range(T-1):
            b = self.emission_prob(observations[t+1])
            numerator = alpha[t].reshape(-1, 1) * self.A * (b * beta[t+1]).reshape(1, -1)
            xi[t] = numerator / np.sum(numerator)

        return xi

    def m_step(self, observations, gamma, xi):
        T = len(observations)
        alpha = self.forward(observations)
        beta = self.backward(observations)

        # update pi
        self.pi = gamma[0]

        # update A
        self.A = xi.sum(axis=0) / gamma[:-1].sum(axis=0, keepdims=True).T

        # update mu and sigma
        for k in range(self.n_states):
            gamma_k = gamma[:, k]  # shape (T,) — weight for state k at each timestep
            
            # update mu
            self.mu[k] = np.sum(gamma_k.reshape(-1,1) * observations, axis=0) / np.sum(gamma_k)

            # update sigma
            diff = observations - self.mu[k]  # shape (T, n_feats)
            self.sigma[k] = (gamma_k.reshape(-1,1) * diff).T @ diff / np.sum(gamma_k)

    def fit(self, observations, n_iter=100, tol=1e-4):
        prev_log_likelihood = None

        for i in range(n_iter):
            # E step
            alpha = self.forward(observations)
            beta = self.backward(observations)
            gamma = self.compute_gamma(alpha, beta)
            xi = self.compute_xi(observations, alpha, beta)

            # M step
            self.m_step(observations, gamma, xi)

            # check convergence
            log_likelihood = np.log(alpha[-1].sum())
            if prev_log_likelihood is not None:
                if abs(log_likelihood - prev_log_likelihood) < tol:
                    print(f"Converged at iteration {i}")
                    break
            prev_log_likelihood = log_likelihood


    def predict_regime(self, observations):
        alpha = self.forward(observations)
        probs = alpha[-1] / alpha[-1].sum()
        return np.argmax(probs)
