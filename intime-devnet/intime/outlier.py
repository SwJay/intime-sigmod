import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import math
import logging

import numpy as np
from scipy.stats import lognorm, norm
from statsmodels.stats.stattools import medcouple

import lmoments3 as lm

import agnostic


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_dir = 'intime_logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(f'{log_dir}/outlier.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


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
    logger.info(f"dtmad_ol 保留了 {len(x)}/{len(data)} 个数据点")
    
    return x, pred_mask

# shift estimation ====================

def lmom_est(data):    
    lmom = lm.lmom_ratios(data, nmom=3)
    l1, l2, t3 = lmom

    logger.info(f"lmom: {l1}, {l2}, {t3}")
    t3 = max(t3, 1e-3) # positive skew

    z = np.sqrt(8 / 3) * norm.ppf((1 + t3) / 2)
    sigma = 0.999281 * z - 0.006118 * z**3 + 0.000127 * z**5

    mu = np.log(l2 / math.erf(sigma / 2)) - sigma**2 / 2
    zeta = l1 - np.exp(mu + sigma**2 / 2)
    
    # 检查最终结果是否为 nan
    if np.isnan(mu) or np.isnan(sigma) or np.isnan(zeta):
        logger.warning(f"计算结果包含 NaN: mu: {mu}, sigma: {sigma}, zeta: {zeta}")
    else:
        logger.info(f"计算结果: mu: {mu}, sigma: {sigma}, zeta: {zeta}")
    
    return mu, sigma, zeta

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
