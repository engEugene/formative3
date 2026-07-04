import numpy as np

def gaussian_pdf(x, mu, var):
    """Calculates the Probability Density Function of a Gaussian distribution."""
    eps = 1e-8 # Tiny number to prevent division by zero
    coeff = 1.0 / np.sqrt(2 * np.pi * (var + eps))
    exponent = np.exp(-0.5 * ((x - mu) ** 2) / (var + eps))
    return coeff * exponent

def calculate_log_likelihood(X, mu1, mu2, var1, var2, pi1, pi2):
    """Calculates the total log-likelihood of the data given current parameters."""
    p1 = pi1 * gaussian_pdf(X, mu1, var1)
    p2 = pi2 * gaussian_pdf(X, mu2, var2)
    # 1e-300 prevents log(0) in case of extreme floating point underflow
    return np.sum(np.log(p1 + p2 + 1e-300))