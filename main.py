# script used for running the latent structure / EM algorithm
from expectationmaximization import ExpectationMaximization

# list our columns of interest
base_sfx = [f"Ex1_Side{i + 1}a" for i in range(25) if i + 1 not in [1, 11, 19, 23, 24, 25]]
mid_sfx = [f"Ex2_Side{i + 1}a" for i in range(25) if i + 1 not in [1, 11, 19, 23, 24, 25]]
end_sfx = [f"Ex3_Side{i + 1}a" for i in range(25) if i + 1 not in [1, 11, 19, 23, 24, 25]]

# specify which we want to fit a basis for, and which we should fit xs for after
fit_cols = base_sfx
# note that this should be a list-of-lists for each time point!
delta_cols = [mid_sfx, end_sfx]

# specify number of desired dimensions and if centering term should be included
with_center = True
d = 2

# specify path to csv, directory for basis to be saved, and path to save result
data_file = "data.csv"
basis_dir = "."
result_path = "output.csv"
# specify prefix for new columns
latent_colname = "latent"

# fit model and save latent variables to file
if __name__ == "__main__":
    # create EM object
    em = ExpectationMaximization(data_file=data_file, cols=fit_cols, delta_cols=delta_cols, 
                                with_center=with_center, latent_colname=latent_colname)
    # fit basis based on specified object
    fit = em.fit(x_dims=d, save_prefix=basis_dir)
    # save and fit latent variables in format of original dataframe
    latent = em.save_latent(basis_dir, x_dims=d, save_path=result_path)
    # saved dataframe now has columns "latent_..." corresponding to each dimension/measurement!