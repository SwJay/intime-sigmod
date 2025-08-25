import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
import datetime

import numpy as np
from scipy.stats import lognorm
from scipy.special import erfinv
from scipy.spatial.distance import cdist, euclidean
import pandas as pd
from sklearn.metrics import f1_score

import agnostic as agnostic
import outlier
import receipt

from multiprocessing import Pool


# ====================================
# Shift

# ------------------------------------
# outlier removal


def worker_shiftol(args):
    j, m, eta, mu, sigma, shift, mu_ol, sigma_ol = args

    print(f'iter: {j} m: {m} eta: {eta} mu: {mu} sigma: {sigma} shift: {shift} mu_ol: {mu_ol} sigma_ol: {sigma_ol}')

    methods = [outlier.dtmad_ol, outlier.mad_ol, outlier.sn_ol, outlier.iqr_ol, outlier.adj_bxplt_ol]
    
    f1 = np.zeros(len(methods))

    data, true_mask = outlier.gen_data_ln3_1d(m, eta, mu, sigma, shift, mu_ol, sigma_ol)

    clean_datas = [[] for _ in range(len(methods))]
    pred_mask = np.zeros((len(methods), m))

    for i, method in enumerate(methods):
        clean_datas[i], pred_mask[i] = method(data)

        f1[i] = f1_score(true_mask, pred_mask[i])

    return f1


def test_shiftol_eta_pool():
    m = 1000
    mu = 0
    sigma = 1
    shift = 0

    med = lognorm.median(sigma, shift, np.exp(mu))
    std = lognorm.std(sigma, shift, np.exp(mu))

    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    dev_ol = 1
    mu_ol = med - dev_ol * std
    sigma_ol = 1

    iter = 10
    num_methods = 5

    avg_f1s = np.zeros((len(etas), num_methods))

    for i, eta in enumerate(etas):
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, m, eta, mu, sigma, shift, mu_ol, sigma_ol) for j in range(iter)]
            f1 = pool.map(worker_shiftol, args)

        avg_f1s[i] = np.mean(f1, axis=0)

    if not os.path.exists('res/shift/eta'):
        os.makedirs('res/shift/eta')
    np.savetxt('res/shift/eta/f1.csv', avg_f1s, delimiter=',')


def test_shiftol_devol_pool():
    m = 1000
    mu = 0
    sigma = 1
    shift = 0

    med = lognorm.median(sigma, shift, np.exp(mu))
    std = lognorm.std(sigma, shift, np.exp(mu))

    eta = 0.15
    dev_ols = np.array([0.25, 0.5, 1, 2, 4])
    sigma_ol = 1

    iter = 10
    num_methods = 5

    avg_f1s = np.zeros((len(dev_ols), num_methods))

    for i, dev_ol in enumerate(dev_ols):
        mu_ol = med - dev_ol * std
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, m, eta, mu, sigma, shift, mu_ol, sigma_ol) for j in range(iter)]
            f1 = pool.map(worker_shiftol, args)

        avg_f1s[i] = np.mean(f1, axis=0)

    if not os.path.exists('res/shift/dev_ol'):
        os.makedirs('res/shift/dev_ol')
    np.savetxt('res/shift/dev_ol/f1.csv', avg_f1s, delimiter=',')


# ------------------------------------
# shift estimation


def worker_shiftest(args):
    j, m, eta, mu, sigma, shift, mu_ol, sigma_ol = args

    print(f'iter: {j} m: {m} eta: {eta} mu: {mu} sigma: {sigma} shift: {shift} mu_ol: {mu_ol} sigma_ol: {sigma_ol}')

    methods = [outlier.or_lmom_est, outlier.lmom_est, outlier.mmle_est, outlier.pivotal_est]
    
    shift_errs = np.zeros(len(methods))

    data, _ = outlier.gen_data_ln3_1d(m, eta, mu, sigma, shift, mu_ol, sigma_ol)

    for i, method in enumerate(methods):
        params = method(data)
        # print(f'params: {params[i]}')

        shift_errs[i] = np.abs(params[2] - shift)

    return shift_errs


def test_shiftest_eta_pool():
    m = 1000
    mu = 0
    sigma = 1
    shift = 0

    med = lognorm.median(sigma, shift, np.exp(mu))
    std = lognorm.std(sigma, shift, np.exp(mu))

    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    dev_ol = 1
    mu_ol = med - dev_ol * std
    sigma_ol = 1

    iter = 10
    num_methods = 4

    avg_shift_errs = np.zeros((len(etas), num_methods))

    for i, eta in enumerate(etas):
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, m, eta, mu, sigma, shift, mu_ol, sigma_ol) for j in range(iter)]
            shift_errs = pool.map(worker_shiftest, args)

        avg_shift_errs[i] = np.mean(shift_errs, axis=0)

    if not os.path.exists('res/shiftest/eta'):
        os.makedirs('res/shiftest/eta')
    np.savetxt('res/shiftest/eta/avg_shift_errs.csv', avg_shift_errs, delimiter=',')


# ====================================
# Agnostic Estimation


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def worker_normal(args):
    j, n, eta, dev_ol = args
    num_methods = 4
    mean_errs = np.zeros(num_methods)

    print(f'iter: {j} n: {n} eta: {eta} dev_ol: {dev_ol}')
  
    data, mu_true = outlier.gen_data_normal_n(n, eta, dev_ol)

    co_med = np.median(data, axis=0)

    geo_med = geometric_median(data)

    p_med, p_w_med = agnostic.agnostic_mean_G1(data)

    mean_errs[0] = np.linalg.norm(co_med - mu_true, ord=2)
    mean_errs[1] = np.linalg.norm(geo_med - mu_true, ord=2)
    mean_errs[2] = np.linalg.norm(p_med - mu_true, ord=2)
    mean_errs[3] = np.linalg.norm(p_w_med - mu_true, ord=2)

    return mean_errs


def test_normal_n_pool():
    ns = np.array([32, 64, 128, 256, 512])
    eta = 0.15
    dev_ol = 1

    iter = 10
    num_methods = 4

    avg_mean_errs = np.zeros((len(ns), num_methods))
    avg_times = np.zeros((len(ns), num_methods))

    for i, n in enumerate(ns):
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, n, eta, dev_ol) for j in range(iter)]
            mean_errs = pool.map(worker_normal, args)

        avg_mean_errs[i] = np.mean(mean_errs, axis=0)

    if not os.path.exists('res/normal/n'):
        os.makedirs('res/normal/n')
    np.savetxt('res/normal/n/mean_err.csv', avg_mean_errs, delimiter=',')


def test_normal_eta_pool():
    n = 128
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    dev_ol = 1

    num_methods = 4
    iter = 10

    avg_mean_errs = np.zeros((len(etas), num_methods))

    for i, eta in enumerate(etas): 
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, n, eta, dev_ol) for j in range(iter)]
            mean_errs = pool.map(worker_normal, args)

        avg_mean_errs[i] = np.mean(mean_errs, axis=0)

    if not os.path.exists('res/normal/eta'):
        os.makedirs('res/normal/eta')
    np.savetxt(f'res/normal/eta/mean_err.csv', avg_mean_errs, delimiter=',')


def test_normal_dev_pool():
    n = 128
    eta = 0.15
    dev_ols = np.array([0.25, 0.5, 1, 2, 4])

    num_methods = 4
    iter = 10

    avg_mean_errs = np.zeros((len(dev_ols), num_methods))

    for i, dev_ol in enumerate(dev_ols): 
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, n, eta, dev_ol) for j in range(iter)]
            mean_errs = pool.map(worker_normal, args)

        avg_mean_errs[i] = np.mean(mean_errs, axis=0)

    if not os.path.exists('res/normal/dev'):
        os.makedirs('res/normal/dev')
    np.savetxt(f'res/normal/dev/mean_err.csv', avg_mean_errs, delimiter=',')


# ====================================
# InTime

# ------------------------------------
# overall estimation

def worker_ar(args):
    j, border, eta, targets, mu, cov, shift, mu_ol, cov_ol = args

    print(f'iter: {j} border: {border} eta: {eta} targets: {len(targets)} mu_ol: {mu_ol[0]}')

    methods = [outlier.or_lmom_est_n, outlier.sme_est, outlier.lmom_est_n, outlier.mmle_est_n, outlier.pivotal_est_n]

    ar = np.zeros(len(mu))
    for j in range(len(mu)):
        ar[j] = lognorm.cdf(border, np.sqrt(cov[j,j]), shift[j], np.exp(mu[j]))
    
    shifts = np.zeros(len(methods))
    mus = np.zeros(len(methods))
    sigmas = np.zeros(len(methods))
    ars = np.zeros(len(methods))

    target_shifts = np.zeros(len(methods))
    target_mus = np.zeros(len(methods))
    target_sigmas = np.zeros(len(methods))
    target_ars = np.zeros(len(methods))

    data = outlier.gen_data_ln3_n(eta, targets, mu, cov, shift, mu_ol, cov_ol)

    for i, method in enumerate(methods):
        est_mus, est_sigmas, est_shifts = method(data)
        
        est_ars = np.zeros(len(mu))
        for j in range(len(mu)):
            est_ars[j] = lognorm.cdf(border, est_sigmas[j], est_shifts[j], np.exp(est_mus[j]))

        # shift_errs[i] = np.abs(params[i][0] - shift) / np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
        shifts[i] = np.mean(est_shifts)
        mus[i] = np.mean(est_mus)
        sigmas[i] = np.mean(est_sigmas)
        ars[i] = np.mean(est_ars)

        target_shifts[i] = np.mean(est_shifts[targets])
        target_mus[i] = np.mean(est_mus[targets])
        target_sigmas[i] = np.mean(est_sigmas[targets])
        target_ars[i] = np.mean(est_ars[targets])

    return shifts, mus, sigmas, ars, target_shifts, target_mus, target_sigmas, target_ars


def test_ar_eta_pool():
    n = 128
    
    mu = np.ones(n) * 0
    cov = np.eye(n) * 1
    shift = np.ones(n) * 0

    border = 1

    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    target_rate = 0.15
    n_target = int(target_rate * n)
    targets = np.random.choice(n, n_target, replace=False)

    med = np.zeros(n_target)
    std = np.zeros(n_target)
    for i, target in enumerate(targets):
        med[i] = lognorm.median(np.sqrt(cov[target,target]), shift[target], np.exp(mu[target]))
        std[i] = lognorm.std(np.sqrt(cov[target,target]), shift[target], np.exp(mu[target]))
    
    dev_ol = 1
    mu_ol = med - dev_ol * std
    cov_ol = np.eye(n_target)

    iter = 10
    num_methods = 5

    avg_shift_errs = np.zeros((len(etas), num_methods))
    avg_mu_errs = np.zeros((len(etas), num_methods))
    avg_sigma_errs = np.zeros((len(etas), num_methods))
    avg_ar_errs = np.zeros((len(etas), num_methods))

    avg_target_shift_errs = np.zeros((len(etas), num_methods))
    avg_target_mu_errs = np.zeros((len(etas), num_methods))
    avg_target_sigma_errs = np.zeros((len(etas), num_methods))
    avg_target_ar_errs = np.zeros((len(etas), num_methods))

    for i, eta in enumerate(etas):
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, border, eta, targets, mu, cov, shift, mu_ol, cov_ol) for j in range(iter)]
            results = pool.map(worker_ar, args)

        avg_shift_errs[i] = np.mean([result[0] for result in results], axis=0)
        avg_mu_errs[i] = np.mean([result[1] for result in results], axis=0)
        avg_sigma_errs[i] = np.mean([result[2] for result in results], axis=0)
        avg_ar_errs[i] = np.mean([result[3] for result in results], axis=0)

        avg_target_shift_errs[i] = np.mean([result[4] for result in results], axis=0)
        avg_target_mu_errs[i] = np.mean([result[5] for result in results], axis=0)
        avg_target_sigma_errs[i] = np.mean([result[6] for result in results], axis=0)
        avg_target_ar_errs[i] = np.mean([result[7] for result in results], axis=0)

    if not os.path.exists('res/ar/eta'):
        os.makedirs('res/ar/eta')
    np.savetxt('res/ar/eta/avg_shift_errs.csv', avg_shift_errs, delimiter=',')
    np.savetxt('res/ar/eta/avg_mu_errs.csv', avg_mu_errs, delimiter=',')
    np.savetxt('res/ar/eta/avg_sigma_errs.csv', avg_sigma_errs, delimiter=',')
    np.savetxt('res/ar/eta/avg_ar_errs.csv', avg_ar_errs, delimiter=',')
    np.savetxt('res/ar/eta/avg_target_shift_errs.csv', avg_target_shift_errs, delimiter=',')
    np.savetxt('res/ar/eta/avg_target_mu_errs.csv', avg_target_mu_errs, delimiter=',')
    np.savetxt('res/ar/eta/avg_target_sigma_errs.csv', avg_target_sigma_errs, delimiter=',')
    np.savetxt('res/ar/eta/avg_target_ar_errs.csv', avg_target_ar_errs, delimiter=',')


def test_ar_devol_pool():
    n = 128
    
    mu = np.ones(n) * 0
    cov = np.eye(n) * 1
    shift = np.ones(n) * 0

    border = 1

    eta = 0.15
    target_rate = 0.15
    n_target = int(target_rate * n)
    targets = np.random.choice(n, n_target, replace=False)

    med = np.zeros(n_target)
    std = np.zeros(n_target)
    for i, target in enumerate(targets):
        med[i] = lognorm.median(np.sqrt(cov[target,target]), shift[target], np.exp(mu[target]))
        std[i] = lognorm.std(np.sqrt(cov[target,target]), shift[target], np.exp(mu[target]))
    
    dev_ols = np.array([0.25, 0.5, 1, 2, 4])
    
    cov_ol = np.eye(n_target)

    iter = 10
    num_methods = 5

    avg_shift_errs = np.zeros((len(dev_ols), num_methods))
    avg_mu_errs = np.zeros((len(dev_ols), num_methods))
    avg_sigma_errs = np.zeros((len(dev_ols), num_methods))
    avg_ar_errs = np.zeros((len(dev_ols), num_methods))

    avg_target_shift_errs = np.zeros((len(dev_ols), num_methods))
    avg_target_mu_errs = np.zeros((len(dev_ols), num_methods))
    avg_target_sigma_errs = np.zeros((len(dev_ols), num_methods))
    avg_target_ar_errs = np.zeros((len(dev_ols), num_methods))

    for i, dev_ol in enumerate(dev_ols):
        mu_ol = med - dev_ol * std

        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, border, eta, targets, mu, cov, shift, mu_ol, cov_ol) for j in range(iter)]
            results = pool.map(worker_ar, args)

        avg_shift_errs[i] = np.mean([result[0] for result in results], axis=0)
        avg_mu_errs[i] = np.mean([result[1] for result in results], axis=0)
        avg_sigma_errs[i] = np.mean([result[2] for result in results], axis=0)
        avg_ar_errs[i] = np.mean([result[3] for result in results], axis=0)

        avg_target_shift_errs[i] = np.mean([result[4] for result in results], axis=0)
        avg_target_mu_errs[i] = np.mean([result[5] for result in results], axis=0)
        avg_target_sigma_errs[i] = np.mean([result[6] for result in results], axis=0)
        avg_target_ar_errs[i] = np.mean([result[7] for result in results], axis=0)

    if not os.path.exists('res/ar/dev_ol'):
        os.makedirs('res/ar/dev_ol')
    np.savetxt('res/ar/dev_ol/avg_shift_errs.csv', avg_shift_errs, delimiter=',')
    np.savetxt('res/ar/dev_ol/avg_mu_errs.csv', avg_mu_errs, delimiter=',')
    np.savetxt('res/ar/dev_ol/avg_sigma_errs.csv', avg_sigma_errs, delimiter=',')
    np.savetxt('res/ar/dev_ol/avg_ar_errs.csv', avg_ar_errs, delimiter=',')
    np.savetxt('res/ar/dev_ol/avg_target_shift_errs.csv', avg_target_shift_errs, delimiter=',')
    np.savetxt('res/ar/dev_ol/avg_target_mu_errs.csv', avg_target_mu_errs, delimiter=',')
    np.savetxt('res/ar/dev_ol/avg_target_sigma_errs.csv', avg_target_sigma_errs, delimiter=',')
    np.savetxt('res/ar/dev_ol/avg_target_ar_errs.csv', avg_target_ar_errs, delimiter=',')


def test_ar_border_pool():
    n = 128
    
    mu = np.ones(n) * 0
    cov = np.eye(n) * 1
    shift = np.ones(n) * 0

    ps = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    eta = 0.15
    target_rate = 0.15
    n_target = int(target_rate * n)
    targets = np.random.choice(n, n_target, replace=False)

    med = np.zeros(n_target)
    std = np.zeros(n_target)
    for i, target in enumerate(targets):
        med[i] = lognorm.median(np.sqrt(cov[target,target]), shift[target], np.exp(mu[target]))
        std[i] = lognorm.std(np.sqrt(cov[target,target]), shift[target], np.exp(mu[target]))
    
    dev_ol = 1
    mu_ol = med - dev_ol * std
    cov_ol = np.eye(n_target)

    iter = 10
    num_methods = 5

    avg_target_shift_errs = np.zeros((len(ps), num_methods))
    avg_target_mu_errs = np.zeros((len(ps), num_methods))
    avg_target_sigma_errs = np.zeros((len(ps), num_methods))
    avg_target_ar_errs = np.zeros((len(ps), num_methods))

    for i, p in enumerate(ps):
        border = np.exp(0 + np.sqrt(2) * 1 * erfinv(2 * p - 1))
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, border, eta, targets, mu, cov, shift, mu_ol, cov_ol) for j in range(iter)]
            results = pool.map(worker_ar, args)
            
        avg_target_shift_errs[i] = np.mean([result[4] for result in results], axis=0)
        avg_target_mu_errs[i] = np.mean([result[5] for result in results], axis=0)
        avg_target_sigma_errs[i] = np.mean([result[6] for result in results], axis=0)
        avg_target_ar_errs[i] = np.mean([result[7] for result in results], axis=0)

    if not os.path.exists('res/ar/border'):
        os.makedirs('res/ar/border')
    np.savetxt('res/ar/border/avg_target_shift_errs.csv', avg_target_shift_errs, delimiter=',')
    np.savetxt('res/ar/border/avg_target_mu_errs.csv', avg_target_mu_errs, delimiter=',')
    np.savetxt('res/ar/border/avg_target_sigma_errs.csv', avg_target_sigma_errs, delimiter=',')
    np.savetxt('res/ar/border/avg_target_ar_errs.csv', avg_target_ar_errs, delimiter=',')


def worker_ar_t(args):
    j, eta, targets, mu, cov, shift, mu_ol, cov_ol = args

    print(f'iter: {j} eta: {eta} targets: {len(targets)} mu_ol: {mu_ol[0]}')

    methods = [outlier.or_lmom_est_n, outlier.sme_est, outlier.lmom_est_n, outlier.mmle_est_n, outlier.pivotal_est_n]

    times = np.zeros(len(methods))

    data = outlier.gen_data_ln3_n(eta, targets, mu, cov, shift, mu_ol, cov_ol)

    for i, method in enumerate(methods):
        start = time.time()
        est_mus, est_sigmas, est_shifts = method(data)
        end = time.time()

        times[i] = end - start
    
    return times


def test_ar_n_pool():
    ns = np.array([32, 64, 128, 256, 512])

    eta = 0.15
    target_rate = 0.15
    
    dev_ol = 1
    
    iter = 10

    num_methods = 5

    avg_times = np.zeros((len(ns), num_methods))

    for i, n in enumerate(ns):
        mu = np.ones(n) * 0
        cov = np.eye(n) * 1
        shift = np.ones(n) * 0

        n_target = int(target_rate * n)
        targets = np.random.choice(n, n_target, replace=False)

        med = np.zeros(n_target)
        std = np.zeros(n_target)
        for j, target in enumerate(targets):
            med[j] = lognorm.median(np.sqrt(cov[target,target]), shift[target], np.exp(mu[target]))
            std[j] = lognorm.std(np.sqrt(cov[target,target]), shift[target], np.exp(mu[target]))

        mu_ol = med - dev_ol * std
        cov_ol = np.eye(n_target)

        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(k, eta, targets, mu, cov, shift, mu_ol, cov_ol) for k in range(iter)]
            ts = pool.map(worker_ar_t, args)

        avg_times[i] = np.mean(ts, axis=0)


    if not os.path.exists('res/ar/n'):
        os.makedirs('res/ar/n')
    np.savetxt('res/ar/n/times.csv', avg_times, delimiter=',')


# ------------------------------------
# incentive compare

def is_canonical_OIM(t, eta):
    next_slot = 12

    mu, sigma = receipt.compute_blk_param()

    # three-thumb, ensure received by next proposer
    pass_rate = 0.9973

    cdf = lognorm.cdf(next_slot - t, sigma, 0, np.exp(mu))

    if cdf >= pass_rate:
        return 1
    else:
        return 0

def is_canonical(t, eta):
    # deadline 4
    deadline = 4
    # ratio
    pass_rate = 0.4

    # # assume block propagation same as txn
    # mu = 0
    # sigma = 1

    mu, sigma = receipt.compute_blk_param()

    if deadline - t <= 0:
        cdf = 0
    else:
        cdf = lognorm.cdf(deadline - t, sigma, 0, np.exp(mu))

    if (cdf + eta) >= pass_rate:
        return 1
    else:
        return 0


def vote(t, eta):
    # deadline 4
    deadline = 4

    # # assume block propagation same as txn
    # mu = 0
    # sigma = 1

    mu, sigma = receipt.compute_blk_param()

    if deadline - t <= 0:
        cdf = 0
    else:
        cdf = lognorm.cdf(deadline - t, sigma, 0, np.exp(mu))

    return min(cdf + eta, 1)


def ari_revenue(fees, mus, sigmas, shifts, label):
    n = len(fees)
    ars = np.zeros(n)
    start = 0
    end = 12

    for i in range(n):
        if label == 'pre':
            ars[i] = 1 - lognorm.cdf(end, sigmas[i], shifts[i], np.exp(mus[i]))
        elif label == 'cur':
            ars[i] = lognorm.cdf(end, sigmas[i], shifts[i], np.exp(mus[i])) - lognorm.cdf(start, sigmas[i], shifts[i], np.exp(mus[i]))
        elif label == 'nex':
            ars[i] = lognorm.cdf(start, sigmas[i], shifts[i], np.exp(mus[i]))

    r = np.dot(ars, fees)
    return r


def worker_t_intime(args):
    j, t, eta = args

    b_num = j + 19000000
    print(f'block: {b_num} t: {t} eta: {eta}')

    num_methods = 4
    rhos = np.zeros(num_methods)

    if t == 0:
        return rhos

    RB = 0.034 * 1e18

    path_cur = f'data/rec_r0/B_{b_num}.csv'
    txns_cur = pd.read_csv(path_cur)
    fees_cur = txns_cur['earn_txn_fee']
    R_0 = fees_cur.sum()

    path_nex = f'data/rec_r0/B_{b_num+1}.csv'
    txns_nex = pd.read_csv(path_nex)
    extra_txns_nex = txns_nex[txns_nex['init_time'] <= t]
    R_extra = extra_txns_nex['earn_txn_fee'].sum()

    # original ===============================
    rhos[0] = max((is_canonical_OIM(t, eta) * R_extra - (1 - is_canonical_OIM(t, eta)) * RB) / R_0, 0)

    # fork ===============================
    rhos[1] = max((is_canonical(t, eta) * R_extra - (1 - is_canonical(t, eta)) * RB) / R_0, 0)

    # RPR ===============================
    rhos[2] = max((vote(t, eta) * R_extra - (1 - vote(t, eta)) * RB) / R_0, 0)

    # ARI ===============================
    ## R_0_ari
    # R_pre = 0
    R_pre_est = 0

    ### pre
    if j > 0:
        path_pre = f'data/rec_r0/B_{b_num-1}.csv'
        txns_pre = pd.read_csv(path_pre)
        R_pre_est = ari_revenue(txns_pre['earn_txn_fee'].values, txns_pre['est_mu'].values, txns_pre['est_sigma'].values, txns_pre['est_shift'].values, 'pre')
    
    # R_0_ari = RB + R_pre
    R_0_ari_est = R_pre_est


    ### cur
    R_0_ari_est += ari_revenue(fees_cur.values, txns_cur['est_mu'].values, txns_cur['est_sigma'].values, txns_cur['est_shift'].values, 'cur')
    
    ### nex
    R_0_ari_est += ari_revenue(txns_nex['earn_txn_fee'].values, txns_nex['est_mu'].values, txns_nex['est_sigma'].values, txns_nex['est_shift'].values, 'nex')

    ## R_t_ari 
    ### pre
    R_t_ari_est = R_pre_est

    ### cur
    # merge art_cur and the first l cols of art_nex
    l = extra_txns_nex.shape[0]

    if l != 0:

        # merge fee, mu, sigma, init_time
        merge_fee_cur = pd.concat([fees_cur, extra_txns_nex['earn_txn_fee']]).values
        merge_init_time_cur = pd.concat([txns_cur['init_time'], extra_txns_nex['init_time']]).values
        merge_mu_cur = pd.concat([txns_cur['mu'], extra_txns_nex['mu']]).values
        merge_sigma_cur = pd.concat([txns_cur['sigma'], extra_txns_nex['sigma']]).values
        merge_init_time_cur[-l:] += 12

        targets = np.arange(merge_fee_cur.shape[0])[-l:]

        _, sigma = receipt.compute_blk_param()
        mu = receipt.compute_txn_param(sigma)

        std = lognorm.std(sigma, 0, np.exp(mu))

        mu_ol = np.ones(l) * 12 - std
        covs_ol = np.eye(len(targets)) * std ** 2

        data = outlier.gen_data_ln3_n(eta, targets, merge_mu_cur, np.diag(merge_sigma_cur ** 2), merge_init_time_cur, mu_ol, covs_ol)
        est_mus, est_sigmas, est_shifts = outlier.sme_est(data)

        R_t_ari_est += ari_revenue(merge_fee_cur, est_mus, est_sigmas, est_shifts, 'cur')
        
        ### nex
        left_txns_nex = txns_nex[txns_nex['init_time'] > t]

        left_fee_nex = left_txns_nex['earn_txn_fee'].values
        left_mu_nex = left_txns_nex['est_mu'].values
        left_sigma_nex = left_txns_nex['est_sigma'].values
        left_shift_nex = left_txns_nex['est_shift'].values

        R_t_ari_est += ari_revenue(left_fee_nex, left_mu_nex, left_sigma_nex, left_shift_nex, 'nex')

    else:
        R_t_ari_est = R_0_ari_est

    # estiamted ARI
    rhos[3] = (R_t_ari_est - R_0_ari_est) / R_0_ari_est

    return rhos


def test_t_intime():
    ts = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    eta = 0.15
    num_blocks = 100
    
    num_methods = 4

    avg_rhos = np.zeros((len(ts), num_methods))
    # max_rhos = np.zeros((len(ts), num_methods))

    for i, t in enumerate(ts): 
        with Pool(processes=int(os.cpu_count() / 2)) as pool:
            args = [(j, t, eta) for j in range(num_blocks)]
            rhos = pool.map(worker_t_intime, args)

        avg_rhos[i] = np.mean(rhos, axis=0)
        # max_rhos[i] = np.max(rhos, axis=0)
    
    # print(max_rhos)

    if not os.path.exists('res/intime'):
        os.makedirs('res/intime')
    np.savetxt('res/intime/t_rho.csv', avg_rhos, delimiter=',')


def worker_eta_intime(args):
    j, ts, eta = args

    b_num = j + 19000000
    # print(f'block: {b_num} ts: {ts} eta: {eta}')

    num_methods = 4
    rhos = np.zeros(num_methods)

    RB = 0.034 * 1e18

    path_cur = f'data/rec_r0/B_{b_num}.csv'
    txns_cur = pd.read_csv(path_cur)
    fees_cur = txns_cur['earn_txn_fee']
    R_0 = fees_cur.sum()

    path_nex = f'data/rec_r0/B_{b_num+1}.csv'
    txns_nex = pd.read_csv(path_nex)
    extra_txns_nex_0 = txns_nex[txns_nex['init_time'] <= ts[0]]
    extra_txns_nex_1 = txns_nex[txns_nex['init_time'] <= ts[1]]
    extra_txns_nex_2 = txns_nex[txns_nex['init_time'] <= ts[2]]
    extra_txns_nex_3 = txns_nex[txns_nex['init_time'] <= ts[3]]
    
    R_extra_0 = extra_txns_nex_0['earn_txn_fee'].sum()
    R_extra_1 = extra_txns_nex_1['earn_txn_fee'].sum()
    R_extra_2 = extra_txns_nex_2['earn_txn_fee'].sum()

    # original ===============================
    rhos[0] = max((is_canonical_OIM(ts[0], eta) * R_extra_0 - (1 - is_canonical_OIM(ts[0], eta)) * RB) / R_0, 0)

    # fork ===============================
    rhos[1] = max((is_canonical(ts[1], eta) * R_extra_1 - (1 - is_canonical(ts[1], eta)) * RB) / R_0, 0)

    # RPR ===============================
    rhos[2] = max((vote(ts[2], eta) * R_extra_2 - (1 - vote(ts[2], eta)) * RB) / R_0, 0)

    # ARI ===============================
    ## R_0_ari
    # R_pre = 0
    R_pre_est = 0

    ### pre
    if j > 0:
        path_pre = f'data/rec_r0/B_{b_num-1}.csv'
        txns_pre = pd.read_csv(path_pre)
        R_pre_est = ari_revenue(txns_pre['earn_txn_fee'].values, txns_pre['est_mu'].values, txns_pre['est_sigma'].values, txns_pre['est_shift'].values, 'pre')
    
    # R_0_ari = RB + R_pre
    R_0_ari_est = R_pre_est

    ### cur
    R_0_ari_est += ari_revenue(fees_cur.values, txns_cur['est_mu'].values, txns_cur['est_sigma'].values, txns_cur['est_shift'].values, 'cur')
    
    ### nex
    R_0_ari_est += ari_revenue(txns_nex['earn_txn_fee'].values, txns_nex['est_mu'].values, txns_nex['est_sigma'].values, txns_nex['est_shift'].values, 'nex')

    ## R_t_ari 
    ### pre
    R_t_ari_est = R_pre_est

    ### cur
    # merge art_cur and the first l cols of art_nex
    l = extra_txns_nex_3.shape[0]

    if l != 0:

        # merge fee, mu, sigma, init_time
        merge_fee_cur = pd.concat([fees_cur, extra_txns_nex_3['earn_txn_fee']]).values
        merge_init_time_cur = pd.concat([txns_cur['init_time'], extra_txns_nex_3['init_time']]).values
        merge_mu_cur = pd.concat([txns_cur['mu'], extra_txns_nex_3['mu']]).values
        merge_sigma_cur = pd.concat([txns_cur['sigma'], extra_txns_nex_3['sigma']]).values
        merge_init_time_cur[-l:] += 12

        targets = np.arange(merge_fee_cur.shape[0])[-l:]

        _, sigma = receipt.compute_blk_param()
        mu = receipt.compute_txn_param(sigma)

        std = lognorm.std(sigma, 0, np.exp(mu))

        mu_ol = np.ones(l) * 12 - std
        covs_ol = np.eye(len(targets)) * std ** 2

        data = outlier.gen_data_ln3_n(eta, targets, merge_mu_cur, np.diag(merge_sigma_cur ** 2), merge_init_time_cur, mu_ol, covs_ol)
        est_mus, est_sigmas, est_shifts = outlier.sme_est(data)

        R_t_ari_est += ari_revenue(merge_fee_cur, est_mus, est_sigmas, est_shifts, 'cur')
        
        ### nex
        left_txns_nex = txns_nex[txns_nex['init_time'] > ts[3]]

        left_fee_nex = left_txns_nex['earn_txn_fee'].values
        left_mu_nex = left_txns_nex['est_mu'].values
        left_sigma_nex = left_txns_nex['est_sigma'].values
        left_shift_nex = left_txns_nex['est_shift'].values

        R_t_ari_est += ari_revenue(left_fee_nex, left_mu_nex, left_sigma_nex, left_shift_nex, 'nex')

    else:
        R_t_ari_est = R_0_ari_est

    # estiamted ARI
    rhos[3] = (R_t_ari_est - R_0_ari_est) / R_0_ari_est

    return rhos


def test_eta_intime():
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    # etas = np.array([0, 0.1, 0.2, 0.3])
    num_blocks = 100
    iters = 10
    epsilon = 1e-3
    
    num_methods = 4

    d1_rhos = np.zeros(num_methods)
    t_rhos = np.zeros(num_methods)
    d2_rhos = np.zeros(num_methods)

    thresholds = np.zeros((len(etas), num_methods))

    for i, eta in enumerate(etas):
        print(f'======= {eta} =======')
        a_s = np.zeros(num_methods)
        b_s = np.ones(num_methods) * 12
        t_s = np.ones(num_methods) * 6
        step = 3
        d1_s = t_s - step
        d2_s = t_s + step
        
        for j in range(iters):
            print(f'======= {j} =======')
            with Pool(processes=int(os.cpu_count() / 2)) as pool:
                args = [(k, d1_s, eta) for k in range(num_blocks)]
                rhos = pool.map(worker_eta_intime, args) 
            d1_rhos = np.mean(rhos, axis=0)

            with Pool(processes=int(os.cpu_count() / 2)) as pool:
                args = [(k, t_s, eta) for k in range(num_blocks)]
                rhos = pool.map(worker_eta_intime, args) 
            t_rhos = np.mean(rhos, axis=0)

            with Pool(processes=int(os.cpu_count() / 2)) as pool:
                args = [(k, d2_s, eta) for k in range(num_blocks)]
                rhos = pool.map(worker_eta_intime, args) 
            d2_rhos = np.mean(rhos, axis=0)
            
            step /= 2
            for l in range(num_methods):
                if d1_rhos[l] + epsilon < t_rhos[l] and t_rhos[l] + epsilon < d2_rhos[l]:
                    a_s[l] = t_s[l]
                elif d1_rhos[l] > t_rhos[l] + epsilon and t_rhos[l] > d2_rhos[l] + epsilon:
                    b_s[l] = t_s[l]
                elif d1_rhos[l] + epsilon < t_rhos[l] and t_rhos[l] > d2_rhos[l] + epsilon:
                    a_s[l] = d1_s[l]
                    b_s[l] = d2_s[l]
                else:
                    b_s[l] = t_s[l]
            
            t_s = (a_s + b_s) / 2
            d1_s = t_s - step
            d2_s = t_s + step
        
        thresholds[i] = t_s

    if not os.path.exists('res/intime'):
        os.makedirs('res/intime')
    np.savetxt('res/intime/eta_thre.csv', thresholds, delimiter=',')


def test_eta_predict():
    etas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    delays_csv = 'res/intime/eta_thre.csv'

    delays = np.loadtxt(delays_csv, delimiter=',')

    num_blocks = 100
    num_methods = 4
    iters = 100

    mu, sigma = receipt.compute_blk_param()

    avg_mins = np.zeros((len(etas), num_methods))
    avg_q1s = np.zeros((len(etas), num_methods))
    avg_meds = np.zeros((len(etas), num_methods))
    avg_q3s = np.zeros((len(etas), num_methods))
    avg_maxs = np.zeros((len(etas), num_methods))
    avg_means = np.zeros((len(etas), num_methods))
    avg_vars = np.zeros((len(etas), num_methods))

    for i, eta in enumerate(etas):
        delay = delays[i]

        mins = np.zeros((iters, num_methods))
        q1s = np.zeros((iters, num_methods))
        meds = np.zeros((iters, num_methods))
        q3s = np.zeros((iters, num_methods))
        maxs = np.zeros((iters, num_methods))
        means = np.zeros((iters, num_methods))
        vars = np.zeros((iters, num_methods))

        for iter in range(iters):
            # block process time
            blk_ts = lognorm.rvs(s=sigma, loc=12, scale=np.exp(mu), size=num_blocks)
            
            # index of malicious delay block
            idx = np.random.choice(num_blocks, int(eta * num_blocks), replace=False)

            for j in range(num_methods):                
                blk_ts_ = blk_ts.copy()
                blk_ts_[idx] += delay[j]

                mins[iter][j] = np.min(blk_ts_)
                q1s[iter][j] = np.quantile(blk_ts_, 0.25)
                meds[iter][j] = np.median(blk_ts_)
                q3s[iter][j] = np.quantile(blk_ts_, 0.75)
                maxs[iter][j] = np.max(blk_ts_)
                means[iter][j] = np.mean(blk_ts_)
                vars[iter][j] = np.var(blk_ts_)
        
        avg_mins[i] = np.mean(mins, axis=0)
        avg_q1s[i] = np.mean(q1s, axis=0)
        avg_meds[i] = np.mean(meds, axis=0)
        avg_q3s[i] = np.mean(q3s, axis=0)
        avg_maxs[i] = np.mean(maxs, axis=0)
        avg_means[i] = np.mean(means, axis=0)
        avg_vars[i] = np.mean(vars, axis=0)

    if not os.path.exists('res/predict'):
        os.makedirs('res/predict')
    np.savetxt('res/predict/lat_min.csv', avg_mins, delimiter=',')
    np.savetxt('res/predict/lat_q1.csv', avg_q1s, delimiter=',')
    np.savetxt('res/predict/lat_med.csv', avg_meds, delimiter=',')
    np.savetxt('res/predict/lat_q3.csv', avg_q3s, delimiter=',')
    np.savetxt('res/predict/lat_max.csv', avg_maxs, delimiter=',')
    np.savetxt('res/predict/lat_mean.csv', avg_means, delimiter=',')
    np.savetxt('res/predict/lat_var.csv', avg_vars, delimiter=',')



if __name__ == '__main__':
    # os.chdir(os.path.abspath('code/intime'))
    # print start time
    print(f'Starts: {datetime.datetime.now()}')

    start = time.time()

    # ====================================
    ############# Shift
    
    # ------------------------------------
    ####### outlier removal

    # test_shiftol_eta_pool()
    # test_shiftol_devol_pool()
    
    # ------------------------------------
    ####### shift estimation
    
    # test_shiftest_eta_pool()

    # ====================================
    #############  Agnostic Estimation
    
    # test_normal_n_pool()
    # test_normal_eta_pool()
    # test_normal_dev_pool()

    # ====================================
    #############  InTime

    # ------------------------------------
    ####### shift estimation

    # test_ar_eta_pool()
    # test_ar_devol_pool()
    # test_ar_border_pool()
    # test_ar_n_pool()


    # # ------------------------------------
    # ####### incentive comparee

    test_t_intime()
    test_eta_intime()
    test_eta_predict()



    end = time.time()
    print("time: ", end - start)
    print(f'Ends: {datetime.datetime.now()}')
    