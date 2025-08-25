import numpy as np
import math
from scipy.stats import norm
import time

# Algortihm

def quantile(x,q):
    n = len(x)
    y = np.sort(x)
    return(np.interp(q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))

def prctile(x,p):
    return(quantile(x,np.array(p)/100))

def outlier_damping(X):
    m = X.shape[0]
    n = X.shape[1]
    C = 2
    
    a = np.median(X, axis=0)

    s2 = 0
    for i in range(n):
        z = X[:, i] - np.ones(m) * a[i]
        # z_q40, z_q60 = np.percentile(z, [40, 60])
        z_q40 = prctile(z, 40)
        z_q60 = prctile(z, 60)
        std_q40, std_q60 = norm.ppf(0.4), norm.ppf(0.6)
        s2 += ((z_q60 - z_q40) / (std_q60 - std_q40)) ** 2
    s2 *= C

    # compute the weight of each data point
    w = np.zeros(m)
    for i in range(m):
        w[i] = math.exp(- np.sum((X[i, :] - a) ** 2) / s2)
    
    return w


def weighted_median(data, weights):
    """
    Compute the weighted median of a list.

    Parameters:
    data (list): The data points.
    weights (list): The weights of the data points.

    Returns:
    float: The weighted median.
    """
    # Sort the data
    sorted_data, sorted_weights = zip(*sorted(zip(data, weights)))
    total_weight = sum(weights)
    cumulative_weight = 0
    median = None

    for i, weight in enumerate(sorted_weights):
        cumulative_weight += weight
        if cumulative_weight >= total_weight / 2:
            median = sorted_data[i]
            break

    return median


def agnostic_mean_G1(X):
    m = X.shape[0]
    n = X.shape[1]
    
    w = outlier_damping(X)

    if n <= 1:        
        ag_mean = np.median(X, axis=0)
        w_med = weighted_median(X, w)

        return ag_mean, w_med
    
    # compute the weighted covariance matrix
    mu = np.dot(w, X) / m
    B = X - np.tile(mu, (m, 1))
    B = np.multiply(B, np.sqrt(w)[:, np.newaxis])
    S = np.dot(B.T, B) / m

    D, V = np.linalg.eigh(S)
    ascend_idx = np.argsort(D)
    D = D[ascend_idx]
    V = V[:, ascend_idx]

    PW = np.dot(V[:, :n//2], V[:, :n//2].T)
    weightedProjX = np.multiply(np.dot(X, PW), w[:, np.newaxis])
    est1 = np.mean(weightedProjX, axis=0)

    QV = V[:, n//2:]
    est2_m, est2_wm = agnostic_mean_G1(np.dot(X,QV))
    est2_m = np.dot(est2_m, QV.T)
    est2_wm = np.dot(est2_wm, QV.T)
    est_m = est1 + est2_m
    est_wm = est1 + est2_wm

    return est_m, est_wm
