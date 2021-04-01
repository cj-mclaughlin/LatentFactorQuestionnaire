# main class (also being used to run for the moment)

import numpy as np
import pandas as pd
from math import floor

from preprocessing import clean_data, initialize_params
from likelihood import Likelihood
from optimization import M, M_const
import time
from sampling import particle_filter

class ExpectationMaximization:
    def __init__(self, cols, data_file=None, delta_cols=None, with_center=True, drop_thresh=1, latent_colname="latent"):
        """
        param cols: (binary) list of columns to fit model on
        param data_file: path to csv file containing model data
        param delta_cols: (optional) list of lists of column labels columns to fit at subsequent time-points (see README)
        param with_center: whether or not to include an intercept term (default True)
        param drop_thresh: ratio of missing values required to drop a row (default 1 - 100% missing)
        param latent_colname: column prefix for fitted latent variables (default "latent")
        """
        self.df = pd.read_csv(data_file, index_col=0)
        self.cols = cols
        self.delta_cols = delta_cols
        self.with_center = with_center
        self.latent_colname = latent_colname
        # trim data as desired
        self.df = clean_data(self.df, subset=cols, threshold=1)
        self.best_index = None

    def fit(self, x_dims=3, save_prefix="output/", use_prefit_data=False):
        """
        fit model based on start parameters, then find delta Xs for next time point
        param csv_file: path to data file
        param x_dims: how many dimensions of latent structure x to fit
        """
        start_time = time.time()

        # by default, use data from simulation
        # if datafile is provided, we use this
        fit_df = self.prepare_dataframe(self.df, ret_cols=self.cols)

        # initialize model with default parameters
        A, B = initialize_params(fit_df, x_dims)
        # get rid of B if user specified no centering term
        if not self.with_center:
            B = None
        ll = Likelihood(fit_df, x_dims, A, B)

        # if we've already fit the data, and just want to do one step...
        if use_prefit_data:
            return self.one_step(x_dims, save_prefix)

        # format basis term into vector for scipy optimize
        initial_guess = np.hstack([A.flatten(), B.flatten()]) if self.with_center else A.flatten()
        
        # run numerical optimization of model E-step
        print("Starting Expectation Maximization process...")
        M_const(ll.E, initial_guess)

        # save our model parameters
        self.save_model(ll, prefix=save_prefix)

        # check time to run
        total_time = time.time() - start_time
        print(f"Finished Expectation Maximization in {total_time} seconds")

        # select best index
        best_index = self.select_best_index(x_dims, save_prefix)
        print(f"Best model iteration: {best_index}")
        self.best_index = best_index

    def select_best_index(self, x_dims, data_dir):
        """
        select optimal iteration values
        takes model basis with best likelihood
        """
        E = np.load(f"{data_dir}/Es-{x_dims}.npy")
        best_index = np.argmax(E)
        return best_index
        
    def prepare_dataframe(self, df, ret_cols):
        """
        return pandas dataframe only including columns for model fit
        """
        fit_df = df[ret_cols]
        return fit_df

    def save_model(self, ll, prefix=""):
        """
        Save contents of fitted likelihood class
        param sfx_df: pandas dataframe used to fit model
        param ll: fitted likeilihood object
        param prefix: prefix for saved model names
        """
        x_dims = ll.As[0].shape[1]
        np.save(f"{prefix}/As-{x_dims}.npy", np.array(ll.As))
        np.save(f"{prefix}/Bs-{x_dims}.npy", np.array(ll.Bs))
        np.save(f"{prefix}/Xs-{x_dims}.npy", np.array(ll.x_means))
        np.save(f"{prefix}/Es-{x_dims}.npy", np.array(ll.Es))
        np.save(f"{prefix}/y-{x_dims}.npy", np.array(ll.y))

    def save_latent(self, data_dir, x_dims, save_path="output.csv"):
        """
        Compute latent xs from dataset at each given time-point, using a saved fitted basis
        """
        Xs = np.load(f"{data_dir}/Xs-{x_dims}.npy")
        As = np.load(f"{data_dir}/As-{x_dims}.npy")
        if self.with_center:
            Bs = np.load(f"{data_dir}/Bs-{x_dims}.npy")
        else:
            Bs = np.zeros_like(A[:,0])

        latent_df = self.df

        if self.best_index is None:
            print("Must call fit() before computing Xs for later time-points!")
            return

        # save initial xs
        Xf = Xs[self.best_index]
        for d in range(x_dims):
            latent_df[f"{self.latent_colname}_1_x{d+1}"] = Xf[:, d]

        # for each set of following time-points, fit one step of EM algorithm using saved basis
        for column_set_idx in range(len(self.delta_cols)):
            cols = self.delta_cols[column_set_idx]
            fit_df = latent_df[cols]
            ll = Likelihood(fit_df, x_dims, As[self.best_index], Bs[self.best_index])
            xs = particle_filter(ll.sample_likelihood, params=(fit_df, As[self.best_index], Bs[self.best_index]), 
                                n=fit_df.shape[0], d=x_dims)
            xs = np.mean(xs, axis=0)
            for d in range(x_dims):
                latent_df[f"{self.latent_colname}_{2+column_set_idx}_x{d+1}"] = xs[:, d]

        latent_df.to_csv(save_path)
        return latent_df