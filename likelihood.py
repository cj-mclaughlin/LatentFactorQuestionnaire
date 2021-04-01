# likelihood (E-Step) functionality

import numpy as np
from sampling import particle_filter
from copy import deepcopy
import time

class Likelihood():
    def __init__(self, dataframe, d, A, B, penalty_coef=50):
        """
        constructor
        param dataframe: preprocessed dataset
        param d: desired dimension of X
        param A: initial matrix A (AX + B)
        param B: initial vector B (AX + B)
        param penalty_coef: scalar coefficient for orthogonality penalty
        """
        self.dataframe = dataframe
        self.penalty_coef = penalty_coef
        self.d = d
        self.n, self.m = dataframe.shape
        self.y = dataframe.values
        self.A = A
        self.B = B
        self.with_centering = True
        if B is None:
            # make 0 vector to keep calculations constant
            self.B = np.zeros(shape=(self.m, 1))
            self.with_centering = False
        # keep a history of our parameters over iteration
        self.As = []
        self.Bs = []
        self.x_means = []
        self.Es = []
        # penalty term may on iteration
        self.iteration = 1
        # number of resamples for E-step (effects on iteration time)
        self.n_resamples = 1000

    def sample_likelihood(self, A, B, samples):
        """
        Compute likelihood of samples of x, for use in particle filter
        """
        # will need to replace prob_matrix with 1 where observation is na
        nan_mask = np.ma.masked_invalid(self.y).mask
        # will need to replace prob_matrix with 1-prob where observation is 0
        zeroes_mask = np.ma.masked_where(np.equal(0, self.y), self.y).mask
        prob_matrix = (np.matmul(A, samples.T) + B).T
        prob_matrix = np.exp(prob_matrix) / (1 + np.exp(prob_matrix))
        prob_matrix[nan_mask] = 1
        prob_matrix[zeroes_mask] = 1 - prob_matrix[zeroes_mask]
        return np.product(prob_matrix, axis=1)

    def E(self, params):
        """
        final E-step, calculates full log likelihood
        use parameters A, B as passed in from optimization M-step in params object
        updates history of parameters A, B, X, as well as history of log likelihood
        visualize heatmap of model probability y
        """
        E = 0
        # parse optimization parameter vector into As and Bs, add to model history
        if self.with_centering:
            self.A = params[:len(params) // (self.d + 1) * self.d].reshape(self.m, self.d)
        else:
            self.A = params.reshape(self.m, self.d)

        # normalize
        for i in range(self.d):
            self.A[:,i] = self.A[:,i] / np.linalg.norm(self.A[:,i])
        self.As.append(deepcopy(self.A))

        # check first if we are using a centering term
        if self.with_centering:
            self.B = params[len(params) // (self.d + 1) * self.d:].reshape(self.m, 1)
            self.Bs.append(deepcopy(self.B))

        xs = particle_filter(self.sample_likelihood, params=(self.y, self.A, self.B),
                             n=self.n, d=self.d, n_resamples=self.n_resamples)

        self.x_means.append(np.mean(xs, axis=0))

        for x in xs:
            logit = (np.matmul(self.A, x.T) + self.B).T
            # replace missing by 0, we do not include in summation
            y = np.nan_to_num(self.y, nan=0)
            e_step = np.multiply(y, logit) - np.log(1 + np.exp(logit))
            E += np.sum(e_step)
        # normalize total likelihood to average likelihood per sample
        norm_likelihood = E / self.n_resamples
        self.Es.append(norm_likelihood)
        # compute cost, note likelihood must be negative as we are using a minimizer
        penalty = self.penalty()
        cost = -norm_likelihood + penalty
        print(f"Iteration: {self.iteration}")
        self.iteration += 1
        return cost

    def penalty(self):
        identity = np.identity(self.m)
        # sqrt ( sum ( AAT - I ) ^2 )
        fnorm = np.sqrt(np.sum(np.square(np.matmul(self.A, self.A.T) - identity), axis=None))
        penalty = (self.penalty_coef * fnorm) / self.d
        return penalty