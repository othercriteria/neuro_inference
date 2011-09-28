#!/usr/bin/env python

# Non-clever attempt at inference by using a general-purpose maximizer
# on the log-likelihood.

# Daniel Klein, 9/26/2011

import sys
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat

from utility import window_permutations

# Parameters
params = {'input_file': 'sample.mat',
          'L': 2,
          'Delta': 4}

# Read data from file
input_data = loadmat(params['input_file'])
x = np.asarray(input_data['sample'], dtype='int8')
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
hits = [[]]
s_padded = np.zeros((params['N'],2*params['Delta']), dtype='bool')
for s in windows[0]:
    s_padded[:,params['Delta']:(2*params['Delta'])] = s
    hit = np.empty(theta_dim, dtype='int32')
    for l in range(params['L']):
        t_min, t_max = params['Delta']-(l+1), 2*params['Delta']-(l+1)
        s_lagged = s_padded[:,t_min:t_max]
        hit[:,:,l] = np.tensordot(s_lagged, s, axes = (1,1))
    hits[0].append(hit)
for k in range(1, params['M']):
    hits.append([])
    for s_prev in windows[k-1]:
        hits[k].append([])
        s_padded[:,0:params['Delta']] = s_prev
        for s in windows[k]:
            s_padded[:,params['Delta']:(2*params['Delta'])] = s
            hit = np.empty(theta_dim, dtype='int32')
            for l in range(params['L']):
                t_min, t_max = params['Delta']-(l+1), 2*params['Delta']-(l+1)
                s_lagged = s_padded[:,t_min:t_max]
                hit[:,:,l] = np.tensordot(s_lagged, s, axes = (1,1))
            hits[k][-1].append(hit)
            
# Define objective function, in this case, the negative log-likelihood
def neg_log_likelihood(theta):
    theta = np.reshape(theta, theta_dim)
    h = [np.empty(n_w[0])]
    for w in range(n_w[0]):
        h[0][w] = np.sum(theta * hits[0][w])
    for k in range(1, params['M']):
        h.append(np.empty((n_w[k-1], n_w[k])))
        for w_prev in range(n_w[k-1]):
            for w in range(n_w[k]):
                h[k][w_prev,w] = np.sum(theta * hits[k][w_prev][w])

    b = [np.zeros(n_w[params['M']-1])]            
    for k in range(params['M']-1, 0, -1):
        b = [np.empty(n_w[k-1])] + b
        for w_prev in range(n_w[k-1]):
            for w in range(n_w[k]):
                b[0][w_prev] = np.logaddexp.reduce(h[k][w_prev,:] + b[1])
    b = [None] + b
    log_kappa = np.logaddexp.reduce(h[0] + b[1])

    nll = log_kappa
    nll -= h[0][0]
    for k in range(1, params['M']):
        nll -= h[k][0,0]
    return nll

# Callback for displaying state during optimization
def show_theta(theta):
    theta = np.reshape(theta, (params['N'], params['N'], params['L']))
    if theta_true is None:
        print theta
    else:
        print np.sqrt(np.mean((theta - theta_true)**2))

# Do optimization
print 'Starting optimization'
theta_opt = opt.fmin_bfgs(neg_log_likelihood, theta_init, callback=show_theta)

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
