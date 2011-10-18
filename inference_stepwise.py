#!/usr/bin/env python

# Trying stepwise inference scheme to add non-zero entries to theta
# one at a time.

# Daniel Klein, 10/16/2011

import sys
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from math import factorial

from utility import unlog, fast_average, logaddexp, permute, log_weighted_sample
from utility import theta_viz

# Parameters
profile = True
params = {'input_file': 'EE188_Data.mat',
          'data_field': 'Data',
          'max_T': 25000,
          'max_N': 5,
          'L': 2,
          'perm_max': 4,
          'num_samples': 50,
          'stopping_z': 1.5,
          'lambda': 0.1,
          'opt_params': {'gtol': 0.1, 'maxiter': 5},
          'intermediate_viz': True}

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
        if not (t < params['T'] and i < params['N']): continue
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
    n_perms_memo = {}
    def n_perms(cols):
        n = 0
        denoms = []
        for col in cols:
            n += cols[col]
            denoms.append(cols[col])
        key = (n, tuple(sorted(denoms)))
        if key in n_perms_memo:
            return n_perms_memo[key]
        else:
            val = factorial(n)
            for denom in denoms:
                val /= factorial(denom)
            n_perms_memo[key] = val
            return val
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

    # Initialize theta (using sparse representation)
    theta = {}
    def theta_dense(theta_sparse):
        theta = np.zeros(theta_dim)
        for ind in theta_sparse:
            theta[ind] = theta_sparse[ind]
        return theta
    def arrays_from_theta(theta_sparse):
        inds = []
        theta = []
        for ind in theta_sparse:
            inds.append(ind)
            theta.append(theta_sparse[ind])
        return inds, np.array(theta)
    def theta_from_arrays(inds, vec):
        theta = {}
        for ind, v in zip(inds, vec):
            theta[ind] = v
        return theta

    # Precompute statistics
    print 'Precomputing statistics'
    hits_pre = [np.empty((n_w[0],)+theta_dim)]
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
            hits_pre[0][w,:,:,l] = hit
        if z == window:
            hits_observed += hits_pre[0][w]
    for k in range(1, params['M']):
        hits_pre.append(np.empty((n_w[k-1],n_w[k])+theta_dim))
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
                    hits_pre[k][w_prev,w,:,:,l] = hit
                if z_prev == window_prev and z == window:
                    hits_observed += hits_pre[k][w_prev,w]
    del x_dict

    # Common DP code used for likelihood and gradient calculations
    def dp(theta_sparse):
        theta = theta_dense(theta_sparse)
        
        h = [None] * params['M']
        h[0] = np.empty(n_w[0])
        for w in range(n_w[0]):
            h[0][w] = np.sum(theta * hits_pre[0][w])
        for k in range(1, params['M']):
            h[k] = np.empty((n_w[k-1], n_w[k]))
            for w_prev in range(n_w[k-1]):
                for w in range(n_w[k]):
                    h[k][w_prev,w] = np.sum(theta * hits_pre[k][w_prev,w])

        b = [None] * (params['M']+1)
        b[params['M']] = np.zeros(n_w[params['M']-1])
        for k in range(params['M']-1, 0, -1):
            b[k] = np.empty(n_w[k-1])
            for w_prev in range(n_w[k-1]):
                b[k][w_prev] = logaddexp(h[k][w_prev] + b[k+1])

        return h, b

    # Define objective function, in this case, the negative log-likelihood
    def neg_log_likelihood(theta_vec, inds):
        theta_sparse = theta_from_arrays(inds, theta_vec)
        h, b = dp(theta_sparse)
        
        log_kappa = logaddexp(h[0] + b[1])

        nll = log_kappa
        nll -= h[0][0]
        for k in range(1, params['M']):
            nll -= h[k][0,0]
        nll += params['lambda'] * np.sum(np.abs(theta_vec))
        return nll

    # Compute expected statistics
    def expected_statistics(h, b):
        w_prob = unlog(h[0] + b[1])
        hits_expected = fast_average(hits_pre[0], w_prob)
        for k in range(1, params['M']):
            w_prob_new = np.zeros(n_w[k])
            for w_prev in range(n_w[k-1]):
                w_weight = unlog(h[k][w_prev,:] + b[k+1])
                w_prob_new += w_weight * w_prob[w_prev]
                hits_expected += (w_prob[w_prev] *
                                  fast_average(hits_pre[k][w_prev], w_weight))
            w_prob = w_prob_new
        return hits_expected
    
    # Define gradient of the objective function
    def grad_neg_log_likelihood(theta_vec, inds):
        theta_sparse = theta_from_arrays(inds, theta_vec)
        h, b = dp(theta_sparse)
        
        hits_expected = expected_statistics(h, b)
        grad_full = hits_expected - hits_observed
        
        grad_sparse = []
        for ind in inds:
            grad_sparse.append(grad_full[ind])
        grad_sparse = np.array(grad_sparse)

        # Adjust gradient for L1 regularization
        grad_sparse += params['lambda'] * np.sign(theta_vec)
        
        return grad_sparse

    # Do optimization
    print 'Starting stepwise optimization'
    while True:
        # Visualize current theta
        if params['intermediate_viz']:
            theta_viz(theta_dense(theta))

        # Sample at current theta
        h, b = dp(theta)
        hits_sample = np.zeros((params['num_samples'],)+theta_dim)
        for rep in range(params['num_samples']):
            w_samp = log_weighted_sample(h[0] + b[1])
            hits_sample[rep] += hits_pre[0][w_samp]
            for k in range(1, params['M']):
                w_samp_next = log_weighted_sample(h[k][w_samp,:] + b[k+1])
                hits_sample[rep] += hits_pre[k][w_samp,w_samp_next]
                w_samp = w_samp_next

        # Find component with largest z-score
        hits_expected = expected_statistics(h, b)
        hits_sd = np.sqrt(np.mean((hits_sample-hits_expected)**2, axis=0))
        z_scores = (hits_observed - hits_expected) / hits_sd
        z_scores[hits_sd == 0] = 0
        for ind in theta:
            z_scores[ind] = 0
        argmax_z = np.unravel_index(np.argmax(np.abs(z_scores)), theta_dim)
        if abs(z_scores[argmax_z]) < params['stopping_z']:
            print 'Largest z-score below stopping threshold'
            break
        print 'New component: %s (z = %.2f)' % (str(argmax_z), z_scores[argmax_z])
        theta[argmax_z] = 0.0

        # Refit theta with new non-zero component
        inds, theta_init = arrays_from_theta(theta)
        theta_opt = opt.fmin_bfgs(f = neg_log_likelihood,
                                  #fprime = grad_neg_log_likelihood,
                                  x0 = theta_init,
                                  args = (inds,),
                                  **(params['opt_params']))
        theta = theta_from_arrays(inds, theta_opt)

    # Output
    print 'x'
    print x_sparse
    print

    print 'Parameters'
    for param in params:
        print '%s: %s' % (param, str(params[param]))
    print

    print 'Inferred theta'
    print theta_dense(theta)

if __name__ == '__main__':
    if profile:
        import cProfile, pstats
        cProfile.run('inference(params)', 'inference_stepwise_prof')
        p = pstats.Stats('inference_stepwise_prof')
        p.strip_dirs().sort_stats('time').print_stats(10)
    else:
        inference(params)
