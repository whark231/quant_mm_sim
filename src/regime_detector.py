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

    # Vectorized emission_prob across an entire observation sequence at once.
    # forward()/backward() used to call emission_prob(observations[t]) inside their
    # per-t loops, which costs T*n_states individual scipy calls (fit() calls this
    # every EM iteration) and made training on a realistic multi-thousand-row trade
    # series take minutes. scipy's pdf accepts an array of points directly, so
    # computing one state's density across all T observations in a single call is
    # orders of magnitude faster and gives identical results.
    def emission_matrix(self, observations):
        # observations: shape (T, n_feats); returns B of shape (T, n_states)
        T = len(observations)
        B = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            B[:, k] = multivariate_normal.pdf(observations, mean=self.mu[k], cov=self.sigma[k])
        return B

    # Captures joint probability of being in state k and seeing observations up until time t
    # Scaled per Rabiner (1989): each row of alpha is renormalized to sum to 1 as we go,
    # with the per-step normalizer c[t] stashed away so fit() can recover the true
    # log-likelihood. Without this, alpha underflows to exactly 0.0 after ~100-150 steps
    # (probabilities <1 compounding every t), which silently breaks fit() and
    # predict_regime() on any realistically long observation sequence.
    def forward_scaled(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        scales = np.zeros(T)

        # Guard against an observation so far from every state's mean that all
        # emission densities underflow to exactly 0 (routine once observations
        # are scaler-transformed — plausible under any one state but implausible
        # under all three at once). Dividing by a floored scale alone isn't
        # enough: the row stays all zeros, and since alpha[t] is built from
        # alpha[t-1] @ A, that zero row propagates forever. Falling back to a
        # uniform row keeps every alpha[t] summing to 1 and lets the chain
        # recover once emission densities become informative again.
        eps = np.finfo(float).tiny

        def _normalize(alpha_raw):
            total = alpha_raw.sum()
            if total > eps:
                return alpha_raw / total, total
            return np.full(self.n_states, 1.0 / self.n_states), eps

        # emission_matrix() computes every state's density across the whole
        # sequence in one vectorized scipy call per state, instead of T
        # individual per-timestep calls — fit() runs this every EM iteration,
        # so this is the difference between a walk-forward loop over several
        # folds finishing in seconds vs. tens of minutes.
        B = self.emission_matrix(observations)

        alpha_raw = self.pi * B[0]
        alpha[0], scales[0] = _normalize(alpha_raw)

        for t in range(1, T):
            alpha_raw = B[t] * (alpha[t-1] @ self.A)
            alpha[t], scales[t] = _normalize(alpha_raw)

        return alpha, scales

    # Captures conditional probability of each future observation up until time T given state k at time t
    # Uses the same per-step normalizers c as forward() so alpha/beta stay on a
    # comparable scale (required for compute_gamma/compute_xi to stay correct).
    def backward_scaled(self, observations, scales):
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        beta[T-1] = 1
        B = self.emission_matrix(observations)

        for t in range(T-2, -1, -1):
            beta[t] = (self.A @ (B[t+1] * beta[t+1])) / scales[t]  # use same (already-floored) scale factors as forward

        return beta

    # Result of bayes rule: probability of being in state k given all of the observations
    def compute_gamma(self, alpha, beta):
        # shape: (T, n_states)
        return (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)

    def compute_xi(self, observations, alpha, beta):
        T = len(observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        B = self.emission_matrix(observations)  # same per-EM-iteration bottleneck as forward/backward

        for t in range(T-1):
            b = B[t+1]
            numerator = alpha[t].reshape(-1, 1) * self.A * (b * beta[t+1]).reshape(1, -1)
            xi[t] = numerator / np.sum(numerator)

        return xi

    def m_step(self, observations, gamma, xi):
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
            # small ridge to keep covariance from going singular when a state's
            # effective weight concentrates on too few/collinear observations
            self.sigma[k] += 1e-6 * np.eye(self.n_feats)

    def fit(self, observations, n_iter=100, tol=1e-4):
        prev_log_likelihood = None

        for i in range(n_iter):
            # E step
            alpha, c = self.forward_scaled(observations)
            beta = self.backward_scaled(observations, c)
            gamma = self.compute_gamma(alpha, beta)
            xi = self.compute_xi(observations, alpha, beta)

            # M step
            self.m_step(observations, gamma, xi)

            # check convergence (true log-likelihood recovered from the forward
            # scaling factors: log P(O) = sum_t log(c[t]))
            log_likelihood = np.sum(np.log(c))
            if prev_log_likelihood is not None:
                if abs(log_likelihood - prev_log_likelihood) < tol:
                    print(f"Converged at iteration {i}")
                    break
            prev_log_likelihood = log_likelihood


    def predict_regime(self, observations):
        alpha, _ = self.forward_scaled(observations)
        # forward() already returns rows that sum to 1 (with a uniform
        # fallback on total underflow), so this is a no-op in practice —
        # kept as a defensive floor rather than trusting that invariant blindly.
        total = max(alpha[-1].sum(), np.finfo(float).tiny)
        probs = alpha[-1] / total
        return np.argmax(probs)
