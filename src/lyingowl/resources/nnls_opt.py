import numpy as np
from scipy.optimize import nnls
from scipy.optimize import minimize


def fn(x, A, b):
    return np.linalg.norm(A.dot(x) - b)


def combiner_solve(x, y):
    # adapted from https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function
    # -solution-so-that-it-sums-to-1/33388181
    beta_0, rnorm = nnls(x, y)
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [[0.0, None]] * x.shape[1]
    minout = minimize(fn, beta_0, args=(x, y), method='SLSQP', bounds=bounds, constraints=cons)
    return minout.x  # beta
