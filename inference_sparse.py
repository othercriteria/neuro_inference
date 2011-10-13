#!/usr/bin/env python

# Trying to make inference work on big data without running out of
# memory. The big new idea is adaptive jitter window size.

# Daniel Klein, 10/4/2011

import sys
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from math import factorial

from utility import unlog, fast_average, logaddexp, permute

# Parameters
profile = True
params = {'input_file': 'EE188_Data.mat',
          'data_field': 'Data',
          'max_T': 5000,
          'max_N': 10,
          'L': 4,
          'perm_max': 12,
          'lambda': 0.05}

def inference(params):
    # Read data from file
    input_data = loadmat(params['input_file'])
    x_sparse = np.asarray(input_data[params['data_field']], dtype=np.uint32)
    if 'theta' in input_data:
        theta_true = input_data['theta']
    else:
        theta_true = None
    params['N'], params['T'] = np.max(x_sparse[:,1]), np.max(x_sparse[:,0])
    params['T'] = min(params['T'], params['max_T'])
    params['N'] = min(params['N'], params['max_N'])
    x_sparse -= 1
    theta_dim = (params['N'],params['N'],params['L'])

    # Push sparse data into a dictionary
    print 'Preprocessing sparse data'
    x_dict = {}
    for t in range(params['T']):
        x_dict[t] = []
    for t, i in x_sparse:
        if not t < params['T']: continue
        if not i < params['N']: continue
        x_dict[t].append(i)
    for t in x_dict:
        x_dict[t] = tuple(sorted(x_dict[t]))

    # Define function for building window as needed
    def make_window(cols):
        w = np.zeros((params['N'], len(cols)), dtype=np.bool)
        for o, col in enumerate(cols):
            for i in col:
                w[i, o] = 1
        return w
   
    # Generate S (calling it "windows" in code)
    print 'Counting window permutations'
    def n_perms(cols):
        n = 0
        denom = 1
        for col in cols:
            n += cols[col]
            denom *= factorial(cols[col])
        return factorial(n) / denom
    windows, n_w, l_w = [], [], []
    t_start = 0
    while t_start < params['T']:
        cols_seen = {}
        t_end = t_start
        while t_end < params['T']:
            new_col = x_dict[t_end]
            t_end += 1
            if not new_col in cols_seen:
                cols_seen[new_col] = 0
            cols_seen[new_col] += 1
            if t_end - t_start <= params['L']:
                n_perm = n_perms(cols_seen)
                continue
            n_perm_new = n_perms(cols_seen)
            if n_perm_new > params['perm_max']:
                t_end -= 1
                break
            n_perm = n_perm_new
        windows.append((t_start, t_end))
        n_w.append(n_perm)
        l_w.append(t_end - t_start)
        t_start = t_end
    params['M'] = len(n_w)

    # Initialize theta
    theta_init = np.zeros(theta_dim)

    # Precompute statistics
    print 'Precomputing statistics'
    hits = [np.empty((n_w[0],)+theta_dim)]
    hits_observed = np.zeros(theta_dim)
    s_padded = np.zeros((params['N'],params['L']+l_w[0]), dtype=np.bool)
    w_start, w_end = windows[0]
    window = [x_dict[t] for t in range(w_start, w_end)]
    for w, z in enumerate(permute(window)):
        s = make_window(z)
        s_padded[:,params['L']:(params['L']+l_w[0])] = s
        for l in range(params['L']):
            tmin, tmax = params['L']-(l+1), (params['L']+l_w[0])-(l+1)
            s_lagged = s_padded[:,tmin:tmax]
            hit = np.tensordot(s_lagged, s, axes = (1,1))
            hits[0][w,:,:,l] = hit
        if z == window:
            hits_observed += hits[0][w]
    for k in range(1, params['M']):
        hits.append(np.empty((n_w[k-1],n_w[k])+theta_dim))
        s_padded = np.empty((params['N'],l_w[k-1]+l_w[k]), dtype=np.bool)
        w_prev_start, w_prev_end = windows[k-1]
        w_start, w_end = windows[k]
        window_prev = [x_dict[t] for t in range(w_prev_start, w_prev_end)]
        window = [x_dict[t] for t in range(w_start, w_end)]
        for w_prev, z_prev in enumerate(permute(window_prev)):
            s_prev = make_window(z_prev)
            s_padded[:,0:l_w[k-1]] = s_prev
            for w, z in enumerate(permute(window)):
                s = make_window(z)
                s_padded[:,l_w[k-1]:(l_w[k-1]+l_w[k])] = s
                for l in range(params['L']):
                    tmin, tmax = l_w[k-1]-(l+1), (l_w[k-1]+l_w[k])-(l+1)
                    s_lagged = s_padded[:,tmin:tmax]
                    hit = np.tensordot(s_lagged, s, axes = (1,1))
                    hits[k][w_prev,w,:,:,l] = hit
                if z_prev == window_prev and z == window:
                    hits_observed += hits[k][w_prev,w]

    # Common DP code used for likelihood and gradient calculations
    def dp(theta):
        h = [None] * params['M']
        h[0] = np.empty(n_w[0])
        for w in range(n_w[0]):
            h[0][w] = np.sum(theta * hits[0][w])
        for k in range(1, params['M']):
            h[k] = np.empty((n_w[k-1], n_w[k]))
            for w_prev in range(n_w[k-1]):
                for w in range(n_w[k]):
                    h[k][w_prev,w] = np.sum(theta * hits[k][w_prev,w])

        b = [None] * (params['M']+1)
        b[params['M']] = np.zeros(n_w[params['M']-1])
        for k in range(params['M']-1, 0, -1):
            b[k] = np.empty(n_w[k-1])
            for w_prev in range(n_w[k-1]):
                b[k][w_prev] = logaddexp(h[k][w_prev] + b[k+1])

        return h, b

    # Define objective function, in this case, the negative log-likelihood
    def neg_log_likelihood(theta_vec):
        theta = np.reshape(theta_vec, theta_dim)

        h, b = dp(theta)
        
        log_kappa = logaddexp(h[0] + b[1])

        nll = log_kappa
        nll -= h[0][0]
        for k in range(1, params['M']):
            nll -= h[k][0,0]
        nll += params['lambda'] * np.sum(np.abs(theta))
        return nll
    
    # Define gradient of the objective function
    def grad_neg_log_likelihood(theta_vec):
        theta = np.reshape(theta_vec, theta_dim)

        h, b = dp(theta)

        # Compute expected statistics
        w_prob = unlog(h[0] + b[1])
        hits_expected = fast_average(hits[0], w_prob)
        for k in range(1, params['M']):
            w_prob_new = np.zeros(n_w[k])
            for w_prev in range(n_w[k-1]):
                w_weight = unlog(h[k][w_prev,:] + b[k+1])
                w_prob_new += w_weight * w_prob[w_prev]
                hits_expected += (w_prob[w_prev] *
                                  fast_average(hits[k][w_prev], w_weight))
            w_prob = w_prob_new

        # Adjust gradient for L1 regularization
        reg = params['lambda'] * np.sign(theta)
        
        return np.reshape(hits_expected - hits_observed + reg, theta_vec.shape)

    # Callback for displaying state during optimization
    def show_theta(theta_vec):
        theta = np.reshape(theta_vec, (params['N'], params['N'], params['L']))
        if theta_true is None:
            print np.round(theta, decimals = 2)
        else:
            diff = np.reshape(theta - theta_true, theta_vec.shape)
            print np.sqrt(np.dot(diff, diff))

    # Do optimization
    print 'Starting optimization'
    theta_opt = opt.fmin_bfgs(f = neg_log_likelihood,
                              fprime = grad_neg_log_likelihood,
                              x0 = theta_init,
                              callback = show_theta)

    # Output
    print 'x'
    print x_sparse
    print

    print 'Parameters'
    for param in params:
        print '%s: %s' % (param, str(params[param]))
    print

    print 'Inferred theta'
    print np.reshape(theta_opt, (params['N'], params['N'], params['L']))

if __name__ == '__main__':
    if profile:
        import cProfile, pstats
        cProfile.run('inference(params)', 'inference_sparse_prof')
        p = pstats.Stats('inference_sparse_prof')
        p.strip_dirs().sort_stats('time').print_stats(10)
    else:
        inference(params)
