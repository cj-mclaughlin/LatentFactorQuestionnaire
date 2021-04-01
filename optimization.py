# out-of-box optimization
from scipy.optimize import minimize
from stochopy import optimize
import numpy as np
import math

def M(function, initial_guess):
    """ 
    M-Step
    maximize likelihood function with initial guess A=1s and B=log-odd
    current best performing optimizer is COBYLA
    """
    return minimize(function, initial_guess, method='COBYLA', options={"maxiter": 500, "rhobeg": 0.5})

def M_const(function, initial_guess):
    """
    Experimental optimization with constrained particle swarm
    """
    bounds = [(-5, 5) for i in range(len(initial_guess))]
    return optimize.minimize(fun=function, bounds=bounds, method="pso", options={"maxiter": 100, "seed": 42, "return_all": False})