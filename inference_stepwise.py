#!/usr/bin/env python

# Trying stepwise inference scheme to add non-zero entries to theta
# one at a time.

# Daniel Klein, 10/16/2011

import sys
import numpy as np
import numpy.linalg as la
from scipy.io import loadmat
from scipy.maxentropy import logsumexp
from math import factorial

from utility import unlog, fast_average, permute, log_weighted_sample
from utility import theta_viz

# Parameters
profile = True
params = {'input_file': 'EE188_Data_reordered.mat',
          'data_field': 'Data',
          'label_field': 'cellID',
          'theta_field': 'theta',
          'max_T': 6000000,
          'max_N': 2,
          'L': 2,
          'Delta': 2,
          'num_samples': 20,
          'stopping_global': 0.1,
          'stopping_z': 1.5,
          'step_size': 0.1,
          'opt_tol': 0.1,
          'lambda': 0.05,
          'intermediate_viz': True}

def inference(params):
    # Read data from file
    input_data = loadmat(params['input_file'])
    x_sparse = np.asarray(input_data[params['data_field']], dtype=np.uint32)
    labels, theta_true = None, None
    if params['label_field'] in input_data:
        labels = input_data[params['label_field']][:,0]
    if params['theta_field'] in input_data:
        theta_true = input_data[params['theta_field']]
    params['N'], params['T'] = np.max(x_sparse[:,1]), np.max(x_sparse[:,0])
    params['T'] = min(params['T'], params['max_T'])
    params['N'] = min(params['N'], params['max_N'])
    params['M'] = int(np.ceil(1.0 * params['T'] / params['Delta']))
    params['T'] = params['Delta'] * params['M']
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
    fact_memo = {}
    def fact(n):
        if n in fact_memo:
            return fact_memo[n]
        else:
            val = factorial(n)
            fact_memo[n] = val
            return val
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
            val = fact(n)
            for denom in denoms:
                val /= fact(denom)
            n_perms_memo[key] = val
            return val
    windows, n_w, l_w = [], [], []
    for k in range(params['M']):
        t_start = k * params['Delta']
        t_end = min(params['T'], (k+1) * params['Delta'])
        cols_seen = {}
        for t in range(t_start, t_end):
            new_col = x_dict[t]
            if not new_col in cols_seen:
                cols_seen[new_col] = 0
            cols_seen[new_col] += 1
            n_perm = n_perms(cols_seen)
        windows.append((t_start, t_end))
        n_w.append(n_perm)
        l_w.append(t_end - t_start)

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
    s_padded = np.zeros((params['N'],params['L']+l_w[0]), dtype=np.uint32)
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
        s_padded = np.empty((params['N'],l_w[k-1]+l_w[k]), dtype=np.uint32)
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
                b[k][w_prev] = logsumexp(h[k][w_prev] + b[k+1])

        return h, b

    # Define objective function, in this case, the negative log-likelihood
    def neg_log_likelihood(theta_sparse, hb = None):
        if not hb is None:
            h, b = hb
        else:
            h, b = dp(theta_sparse)
        
        log_kappa = logsumexp(h[0] + b[1])

        nll = log_kappa
        nll -= h[0][0]
        for k in range(1, params['M']):
            nll -= h[k][0,0]
        for ind in theta_sparse:
            nll += params['lambda'] * np.abs(theta_sparse[ind])
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
    def grad_neg_log_likelihood(theta_sparse, hb = None):
        if not hb is None:
            h, b = hb
        else:
            h, b = dp(theta_sparse)
        
        hits_expected = expected_statistics(h, b)
        grad_full = hits_expected - hits_observed
        
        grad_sparse = {}
        for ind in theta_sparse:
            grad_sparse[ind] = grad_full[ind]

        # Adjust gradient for L1 regularization
        for ind in theta_sparse:
            grad_sparse[ind] += params['lambda'] * np.sign(theta_sparse[ind])
        
        return grad_sparse

    # Do optimization
    print 'Starting stepwise optimization'
    h, b = dp(theta)
    nll = neg_log_likelihood(theta, (h, b))
    print 'Initial negative log-likelihood: %.2f' % nll
    while True:
        # Assess model at current theta
        if params['intermediate_viz']:
            theta_viz(theta_dense(theta), labels = labels)

        # Sample at current theta
        hits_sample = np.zeros((params['num_samples'],)+theta_dim)
        w_samps = log_weighted_sample(h[0] + b[1], params['num_samples'])
        for rep in range(params['num_samples']):
            hits_sample[rep] += hits_pre[0][w_samps[rep]]
        for k in range(1, params['M']):
            w_samps_next = []
            w_samps_next_uniques = []
            w_samps_uniques, inds = np.unique(w_samps, return_inverse=True)
            for i, w in enumerate(w_samps_uniques):
                n = len(inds[inds == i])
                w_samps_next_uniques.append(log_weighted_sample(h[k][w]+b[k+1],n))
            for rep in range(params['num_samples']):
                w_samps_next.append(w_samps_next_uniques[inds[rep]].pop())
                hits_sample[rep] += hits_pre[k][w_samps[rep]][w_samps_next[rep]]
            w_samps = w_samps_next

        # Check global goodness-of-fit
        hits_expected = expected_statistics(h, b)
        hits_norms = np.array([la.norm(hits - hits_expected)
                                for hits in hits_sample])
        ext = np.where(la.norm(hits_observed - hits_expected) > hits_norms)[0]
        score = 1.0 * len(ext) / params['num_samples']
        print 'Global score: %.2f' % score
        if score < params['stopping_global']:
            print 'Global goodness-of-fit criterion achieved'
            break

        # Find component with largest z-score
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

        # Make big steps in direction of new theta component
        grad = grad_neg_log_likelihood(theta, (h, b))
        dir_new = -np.sign(grad[argmax_z])
        while True:
            print 'Making big step'
            old_nll = nll
            theta[argmax_z] += dir_new * params['step_size']
            h, b = dp(theta)
            nll = neg_log_likelihood(theta, (h, b))
            print 'Negative log-likelihood: %.2f' % nll
            if nll > old_nll or abs(nll - old_nll) < params['opt_tol']: break

        # Refit theta with new non-zero component
        print 'Optimization by gradient descent'

        while True:
            old_nll = nll
            grad = grad_neg_log_likelihood(theta, (h, b))
            grad_norm = max(la.norm(np.array(grad.values())), 1.0)
            for ind in grad:
                theta[ind] -= (params['step_size'] / grad_norm) * grad[ind]
            h, b = dp(theta)
            nll = neg_log_likelihood(theta, (h, b))
            print 'Negative log-likelihood: %.2f' % nll
            if nll > old_nll: break

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
