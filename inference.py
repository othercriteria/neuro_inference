#!/usr/bin/env python

# Non-clever attempt at inference by using a general-purpose maximizer
# on the log-likelihood.

# Daniel Klein, 9/26/2011

import sys
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat

from utility import window_permutations, unlog, fast_average, logaddexp

# Parameters
profile = True
params = {'input_file': 'sample.mat',
          'L': 2,
          'Delta': 4}

def inference(params):
    # Read data from file
    input_data = loadmat(params['input_file'])
    x = np.asarray(input_data['sample'], dtype=np.uint8)
    if 'theta' in input_data:
        theta_true = input_data['theta']
    else:
        theta_true = None
    params['N'], params['T'] = x.shape
    if not params['T'] % params['Delta'] == 0:
        print 'Error: T must be a multiple of Delta'
        sys.exit()
    params['M'] = params['T'] / params['Delta']
    theta_dim = (params['N'],params['N'],params['L'])
   
    # Generate S (calling it "windows" in code)
    print 'Generating window permutations'
    windows = []
    for k in range(params['M']):
        w = x[:,(k*params['Delta']):((k+1)*params['Delta'])]
        windows.append(window_permutations(w))
    n_w = map(len, windows)

    # Initialize theta
    theta_init = np.zeros(theta_dim)

    # Precompute statistics
    print 'Precomputing statistics'
    hits = [np.empty((n_w[0],)+theta_dim)]
    hits_observed = np.zeros(theta_dim)
    s_padded = np.zeros((params['N'],2*params['Delta']), dtype=np.bool)
    for w, s in enumerate(windows[0]):
        s_padded[:,params['Delta']:(2*params['Delta'])] = s
        for l in range(params['L']):
            tmin, tmax = params['Delta']-(l+1), 2*params['Delta']-(l+1)
            s_lagged = s_padded[:,tmin:tmax]
            hit = np.tensordot(s_lagged, s, axes = (1,1))
            hits[0][w,:,:,l] = hit
    hits_observed += hits[0][0]
    for k in range(1, params['M']):
        hits.append(np.empty((n_w[k-1],n_w[k])+theta_dim))
        for w_prev, s_prev in enumerate(windows[k-1]):
            s_padded[:,0:params['Delta']] = s_prev
            for w, s in enumerate(windows[k]):
                s_padded[:,params['Delta']:(2*params['Delta'])] = s
                for l in range(params['L']):
                    tmin, tmax = params['Delta']-(l+1), 2*params['Delta']-(l+1)
                    s_lagged = s_padded[:,tmin:tmax]
                    hit = np.tensordot(s_lagged, s, axes = (1,1))
                    hits[k][w_prev,w,:,:,l] = hit
        hits_observed += hits[k][0,0]

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

        return np.reshape(hits_expected - hits_observed, theta_vec.shape)

    # Callback for displaying state during optimization
    def show_theta(theta_vec):
        theta = np.reshape(theta_vec, (params['N'], params['N'], params['L']))
        if theta_true is None:
            print theta
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
    print x
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
        cProfile.run('inference(params)', 'inference_prof')
        p = pstats.Stats('inference_prof')
        p.strip_dirs().sort_stats('time').print_stats(10)
    else:
        inference(params)
