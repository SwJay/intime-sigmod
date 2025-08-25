import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import math

import numpy as np
from scipy.stats import lognorm, norm
from statsmodels.stats.stattools import medcouple

import lmoments3 as lm

import agnostic as agnostic


# data generation ====================

def gen_data_normal_n(n, eta, dev_ol):
    m = 1377779 // 32

    mu = np.zeros(n)
    cov = np.eye(n)
    raw = np.random.multivariate_normal(mu, cov, m)

    med = mu
    std = np.sqrt(np.diag(cov))

    mu_m = med - dev_ol * std
    cov_m = np.eye(n)

    idx = np.random.choice(m, int(eta * m), replace=False)
    raw[idx, :] = np.random.multivariate_normal(mu_m, cov_m, int(eta * m))

    med = np.median(raw, axis=0)
    data = raw - np.tile(med, (m, 1))

    mu_true = mu - med
                                                  
    return data, mu_true 


def gen_data_ln3_1d(m, eta, mu, sigma, shift, mu_ol, sigma_ol):
    samples = lognorm.rvs(s=sigma, loc=shift, scale=np.exp(mu), size=m)
    true_mask = np.ones(m)
    idx = []

    if eta:
        idx = np.random.choice(m, int(eta * m), replace=False)
        samples[idx] = np.random.normal(loc=mu_ol, scale=sigma_ol, size=int(eta * m))

    true_mask[idx] = 0

    return samples, true_mask


def gen_data_ln3_n(eta, targets, mu, cov, shift, mu_ol, cov_ol):
    m = 1377779 // 32
    samples = np.random.multivariate_normal(mu, cov, m)
    samples = np.exp(samples)
    samples += shift

    if eta and len(targets):
        idx = np.random.choice(m, int(eta * m), replace=False)
        contamination = np.random.multivariate_normal(mu_ol, cov_ol, int(eta * m))
        for i, row in enumerate(idx):
            samples[row, targets] = contamination[i]

    return samples


# outlier removal ====================

def dtmad_ol(data):
    b = 1.4826
    epsilon = 1e-3
    iters =  2

    x = data.copy()
    pred_mask = np.zeros(len(x))

    for i in range(iters):
        est_shift = np.min(x) - epsilon
        ln_x = np.log(x - est_shift)
        med = np.median(ln_x)

        est_sd = np.median(np.abs(ln_x - med)) * b
        ln_lb = med - est_sd * 3
        ln_ub = med + est_sd * 3
        lb = np.exp(ln_lb) + est_shift
        ub = np.exp(ln_ub) + est_shift

        x = x[(x >= lb) & (x <= ub)]

    pred_mask[(data >= lb) & (data <= ub)] = 1
    
    return x, pred_mask


def mad_ol(data):
    b = 1.4826

    x = data.copy()
    pred_mask = np.zeros(len(x))

    med = np.median(x)
    est_sd = np.median(np.abs(x - med)) * b

    lb = med - est_sd * 3
    ub = med + est_sd * 3

    x = x[(x >= lb) & (x <= ub)]

    pred_mask[(data >= lb) & (data <= ub)] = 1
    
    return x, pred_mask


def sn_ol(data):
    c = 1.1926

    x = data.copy()
    pred_mask = np.zeros(len(x))

    med = np.median(x)
    abs_diffs = np.abs(x[:, None] - x)
    med_abs_diffs = np.median(abs_diffs, axis=1)

    # Calculate median of medians
    est_sd = np.median(med_abs_diffs) * c
    
    lb = med - est_sd * 3
    ub = med + est_sd * 3

    x = x[(x >= lb) & (x <= ub)]

    pred_mask[(data >= lb) & (data <= ub)] = 1
    
    return x, pred_mask


def iqr_ol(data):
    x = data.copy()
    pred_mask = np.zeros(len(x))

    q3, q1 = np.percentile(x, [75 ,25])
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr

    x = x[(x >= lb) & (x <= ub)]

    pred_mask[(data >= lb) & (data <= ub)] = 1
    
    return x, pred_mask


def adj_bxplt_ol(data):
    iters = 1

    x = data.copy()
    pred_mask = np.zeros(len(x))

    mc = medcouple(x)
    q3, q1 = np.percentile(x, [75 ,25])
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr * np.exp(-3.5 * mc)
    ub = q3 + 1.5 * iqr * np.exp(4 * mc)

    x = x[(x >= lb) & (x <= ub)]

    pred_mask[(data >= lb) & (data <= ub)] = 1
    
    return x, pred_mask


# shift estimation ====================


def lmom_est(data):
    lmom = lm.lmom_ratios(data, nmom=3)
    l1, l2, t3 = lmom
    # print(f'l1: {l1}, l2: {l2}, t3: {t3}')

    z = np.sqrt(8 / 3) * norm.ppf((1 + t3) / 2)
    sigma = 0.999281 * z - 0.006118 * z**3 + 0.000127 * z**5
    # if sigma <= 0:
    #     print(sigma)
    mu = np.log(l2 / math.erf(sigma / 2)) - sigma**2 / 2
    zeta = l1 - np.exp(mu + sigma**2 / 2)
    
    return mu, sigma, zeta


def lmom_est_n(data):
    n = data.shape[1]
    mus = np.zeros(n)
    sigmas = np.zeros(n)
    shifts = np.zeros(n)

    for i in range(n):
        mus[i], sigmas[i], shifts[i] = lmom_est(data[:, i])
    
    return mus, sigmas, shifts


def or_lmom_est(data):
    clean, _ = dtmad_ol(data)
    return lmom_est(clean)

# MMLE

def mmle(data, shift):
    r = 3
    data = np.sort(data)
    n = len(data)
    
    t1 = np.log(data[r-1] - shift) - np.sum(np.log(data -  shift)) / n
    kr = norm.ppf(r / (n + 1))
    t2 = np.sum(np.log(data - shift) ** 2) / n - (np.sum(np.log(data - shift)) / n) ** 2

    theta = t1 - kr * np.sqrt(t2)

    return theta


def mmle_est(data):
    n = len(data)
    shift = iter_shift(mmle, data)
    mu = np.sum(np.log(data - shift)) / n
    sigma = np.sqrt(np.sum(np.log(data - shift)**2)/n - (np.sum(np.log(data - shift))/n)**2)
    
    return mu, sigma, shift


def mmle_est_n(data):
    n = data.shape[1]
    mus = np.zeros(n)
    sigmas = np.zeros(n)
    shifts = np.zeros(n)

    for i in range(n):
        mus[i], sigmas[i], shifts[i] = mmle_est(data[:, i])
    
    return mus, sigmas, shifts


# pivotal

def pivotal(data, shift):
    data = np.sort(data)
    n = len(data)
    
    s1 = np.sum(np.log(data[:n//3] - shift)) / (n//3)
    s2 = np.sum(np.log(data[n//3:n - (n//3)] - shift)) / (n  - 2*(n//3))
    s3 = np.sum(np.log(data[n - (n//3):] - shift)) / (n//3)

    pivotal = (s2 - s1) / (s3 - s2)

    return pivotal - 1


def pivotal_est(data):
    n = len(data)
    shift = iter_shift(pivotal, data)
    mu = np.sum(np.log(data - shift)) / n
    sigma = np.sqrt(np.sum((np.log(data - shift) - mu)**2) / n)

    return mu, sigma, shift


def pivotal_est_n(data):
    n = data.shape[1]
    mus = np.zeros(n)
    sigmas = np.zeros(n)
    shifts = np.zeros(n)

    for i in range(n):
        mus[i], sigmas[i], shifts[i] = pivotal_est(data[:, i])
    
    return mus, sigmas, shifts


def iter_shift(method, data):
    iters = 10
    tol = 1e-3

    x0 = np.min(data) - 2e-1
    x1 = np.min(data) - 1e-1
    shift = 0

    # secant method
    for i in range(iters):
        f_x0 = method(data, x0)
        f_x1 = method(data, x1)

        if abs(f_x1) < tol or abs(f_x1 - f_x0) < tol:
            shift = x1
            break

        shift = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        if shift >= min(data):
            shift = min(data) - 1e-3
        
        x0, x1 = x1, shift
    
    return shift


# overall estimation ====================


def _or_lmom_est_n(data):
    m = data.shape[0]
    n = data.shape[1]

    masks = np.zeros((m, n))
    h_threshold = 0.9

    mus = np.zeros(n)
    sigmas = np.zeros(n)
    shifts = np.zeros(n)
    ars = np.zeros(n)

    for i in range(n):
        # _, masks[:,i] = ln3sd_lb(data[:, i])
        clean_data_i, masks[:,i] = dtmad_ol(data[:, i])
        mus[i], sigmas[i], shifts[i] = lmom_est(clean_data_i)

    # all_mask = np.all(masks, axis=1)
    # construct all_mask that the number of non-zero in each row is greater than n * h_threshold
    all_mask = np.sum(masks, axis=1) >= n * h_threshold

    clean_data = data[all_mask]
    
    return clean_data, mus, sigmas, shifts


def or_lmom_est_n(data):
    clean_data, mus, sigmas, shifts = _or_lmom_est_n(data)
    return mus, sigmas, shifts


def get_MAD(data, shifts):
    b = 1.4826
    n = data.shape[1]
    est_sd = np.zeros(n)

    for i in range(n):
        x = data[:, i]
        x = x[x > shifts[i]]

        ln_x = np.log(x - shifts[i])
        med = np.median(ln_x)
        est_sd[i] = np.median(np.abs(ln_x - med)) * b

    return est_sd


def sme_est(data):
    # stage 1: outlier removal + lmom estimation
    clean_data, est_mus, est_sigmas, est_shifts = _or_lmom_est_n(data)

    # stage 2: normal agnostic
    mask = np.all(clean_data > est_shifts, axis=1)
    clean_data = clean_data[mask]

    ln_data = np.log(clean_data - est_shifts)
    _, est_mus_w = agnostic.agnostic_mean_G1(ln_data)

    # get the median of each col of clean_data
    est_shifts_2 = np.median(clean_data, axis=0) - np.exp(est_mus_w)

    est_sigmas_2 = get_MAD(clean_data, est_shifts_2)

    return est_mus_w, est_sigmas_2, est_shifts_2