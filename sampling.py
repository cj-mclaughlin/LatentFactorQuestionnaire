# sampling methods (e.g. particle filter)

import numpy as np
import pandas as pd


def particle_filter(likelihood_fn, params, n, d, mu=0, sigma=1, n_samples=1, n_resamples=1):
    """
    Particle filter method - draw normal samples, weight by fn
    param likelihood_fn: function used to weight drawn samples
    param params: tuple of parameters used for evaluation function (y, A, B)
    param d: number of dimensions of X
    param n: number of observations
    param mu: initial mean of multivariate normal prior (default 0s)
    param sigma: initial covariance (sigma * identity) to draw samples from (default 1; can alter if needed)
    param n_samples: number of initial samples to take/weight from multivariate normal distribution (default 1000)
    param n_resamples: number of resamples of x to return (default 1000)
    return matrix [ n x d ] likelihood xs
    """
    # prior sample parameters
    prior_mu = np.zeros(d)
    prior_cov = sigma * np.identity(d)
    # create dataframe to store results (columns = dimensions of x_i, along with row (patient) index)
    x_cols = [f"x{x_dim + 1}" for x_dim in range(d)]
    columns = ["row_id"] + x_cols + ["L(x)"]
    # unpack parameters
    y, A, B = params
    # tools for building dataframe of results
    row_id_vector = np.arange(0, n, step=1, dtype=np.float64).reshape(-1, 1)
    sample_df = pd.DataFrame(columns=columns)

    for _ in range(n_samples):
        # create n samples, unit weights
        samples = draw_multivariate_samples(prior_mu, prior_cov, n)

        # get probability matrix for our sample, adjusting for our observation
        likelihood = likelihood_fn(A, B, samples)

        # create matrix row id | x samples | likelihood
        data = np.empty(1)
        data = np.append(row_id_vector, samples, axis=1)
        data = np.append(data, likelihood.reshape(-1, 1), axis=1)

        l_df = pd.DataFrame(data=data, columns=columns)

        # append weighted sample to total df
        sample_df = sample_df.append(l_df, ignore_index=True)

    resamples = resample(sample_df, n, d, n_resamples)
    return resamples

def draw_multivariate_samples(mu, cov, n):
    """
    draw samples n x d from normal distribution (mu, sigma)
    param mu: mean of distribution
    param sigma: variance of distribution
    param d: number of columns in desired sample
    param n: number of samples
    """
    return np.random.default_rng().multivariate_normal(mean=mu, cov=cov, size=n)


def resample(df, n, d, n_samples):
    """
    resample from likelihood df weighting by likelihood
    """
    samples = np.zeros((n_samples, n, d), dtype=np.float64)
    cols = [f"x{i + 1}" for i in range(d)]
    df[f"L(x)_norm"] = df[f"L(x)"] / df.groupby('row_id')[f"L(x)"].transform('sum')

    # np.random.choice(x.index, p=x[f"L(x)_norm"], size=n_samples
    # now have sample n x n_samples where each row index has a list of n_samples indices to get xs from

    # x.index["row_index"] = (500,1) gives us the 500 resampled indices for a given row
    # df.iloc[x.index["row_index"]] = (500, 4) gives us the resampled vector x for given row
    sample_indices = df.groupby("row_id").apply(lambda x: np.random.choice(x.index, p=x[f"L(x)_norm"], size=n_samples))

    for i in np.arange(0, n, 1):
        samples[:, i, :] = df.iloc[sample_indices[i]][cols]

    return samples
