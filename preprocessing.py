# preprocessing functionality

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(dataframe, subset, threshold=1):
    """
    drop observations with amount of missing values over a threshold
    param dataframe: dataframe containing binary-only variables
    param subset: subset of data whose columns we are fitting to
    param threshold: ratio of missing values required to drop an observation (default 1 [100% missing])
    returns: dataframe after dropping values
    """
    n_to_drop = (1 - threshold) * dataframe.shape[0]
    return dataframe.dropna(thresh=n_to_drop, subset=subset)


def initialize_params(clean_dataframe, d):
    """
    initialize parameters A and B based on given dataframe
    A is initialized to a random orthogonal matrix
    b is initialized to the log-odds of each term
    param clean_dataframe: preprocessed dataframe
    param d: desired dimension of latent structure X
    """
    # for now A can just be matrix of 1s in proper dimensions, or a random orthonormal matrix ( m x d )
    # we can provide an estimate of B ( m x 1 ) based on our input data, where each column is the log-odds of our y_i
    m = clean_dataframe.shape[1]
    A = np.random.random(size=(m, d))
    A, _ = np.linalg.qr(A)
    B = init_b(clean_dataframe)
    return A, B


def init_b(df):
    """
    initialize b to the log odds given of dataframe
    param df: dataframe
    returns m x 1 vector
    """
    b = []
    for col in df.columns:
        has_sfx = df[col].sum()
        not_sfx = df[col].count() - df[col].sum()
        log_odds = np.log1p(has_sfx / (not_sfx + 1))
        b.append(log_odds)
    return np.array(b, dtype=np.float64)