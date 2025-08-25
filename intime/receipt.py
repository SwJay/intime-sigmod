import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob
import random

import numpy as np
import pandas as pd
from scipy.special import erfinv

from multiprocessing import Pool

import outlier



# real data processing ====================

def distribute_blocks():
    out_dir = 'data/rec_raw'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    blocks_df = pd.read_csv('data/blocks.csv')
    blocks_df = blocks_df[['number', 'base_fee_per_gas']]

    receipts_df = pd.read_csv('data/receipts.csv')
    grouped = receipts_df.groupby('block_number')

    for name, group in grouped:
        group = group.sort_values('transaction_index')
        group = group[['transaction_hash', 'transaction_index', 'block_number', 'gas_used', 'effective_gas_price']]
        
        # Merge the two DataFrames on 'block_number' to append 'base_fee_per_gas'
        merged = pd.merge(group, blocks_df, left_on='block_number', right_on='number', how='left').drop(columns='number')
        merged['earn_txn_fee'] = merged['gas_used'] * (merged['effective_gas_price'] - merged['base_fee_per_gas'])
        
        # name is the block number
        merged.to_csv(f'{out_dir}/B_{name}.csv', index=False)


# generate arrival time & R0 estimation ====================

def worker_gen_time_est(args):
    filename, mu, sigma = args
    print(f'filename: {filename}')

    in_dir = 'data/rec_raw'
    out_dir = 'data/rec_r0'
    
    df = pd.read_csv(os.path.join(in_dir, filename))
    n = len(df)

    eta = 0
    targets = []
    shifts = [random.uniform(0, 12) for _ in range(n)]
    mus = np.ones(n) * mu
    covs = np.eye(n) * sigma ** 2
    devs_ol = []
    covs_ol = []

    data = outlier.gen_data_ln3_n(eta, targets, mus, covs, shifts, devs_ol, covs_ol)
    est_mus, est_sigmas, est_shifts = outlier.sme_est(data)

    df['init_time'] = shifts
    df['mu'] = mus
    df['sigma'] = np.ones(n) * sigma

    df['est_mu'] = est_mus
    df['est_sigma'] = est_sigmas
    df['est_shift'] = est_shifts
    
    df.sort_values('init_time', inplace=True)
    df.to_csv(os.path.join(out_dir, filename), index=False)


# compute arrival rate
def gen_time_est_pool():
    in_dir = 'data/rec_raw'
    out_dir = 'data/rec_r0'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    _, sigma = compute_blk_param()
    mu = compute_txn_param(sigma)

    with Pool(processes=int(os.cpu_count() / 2)) as pool:
        args = [(filename, mu, sigma) for filename in os.listdir(in_dir)]
        pool.map(worker_gen_time_est, args)


def compute_blk_param():
    peak = 0.2
    q90 = 0.85

    c = np.sqrt(2) * erfinv(2 * 0.9 - 1)

    sigma = (np.sqrt(c**2 + 4 * np.log(q90 / peak)) - c) / 2
    mu = np.log(q90) - c * sigma

    return mu, sigma


def compute_txn_param(sigma):
    mean = 0.2
    
    mu = np.log(mean) - sigma**2 / 2
    
    return mu


if __name__ == '__main__':
    # os.chdir(os.path.abspath('intime'))
    
    # real txn fee =======================

    distribute_blocks()

    # generate arrival time ==============

    gen_time_est_pool()

    