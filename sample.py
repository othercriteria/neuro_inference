#!/usr/bin/env python

# Trying to generate samples from the conditional model.

# NOTE: My convention of indexing arrays from 0 does not agree with
# the convention in Matt's notes.

# Daniel Klein, 9/14/2011

import sys
import numpy as np
from scipy.io import savemat
from random import random

from utility import log_weighted_sample, window_permutations

# Parameters
params = {'N': 5,
          'T': 500000,
          'L': 2,
          'Delta': 5,
          'theta_method': ('sparse_unique', {'p': 1.0, 'scale': 3.0}),
          # 'theta_method': ('cascade_2', {'strength': 8.0, 'decay': 0.8}),
          # 'S_method': ('random_uniform', {'p_min': 0.05, 'p_max': 0.2}),
          'S_method': ('random_periodic', {'baseline': 0.005, 'scale': 0.002})}
if not params['T'] % params['Delta'] == 0:
    print 'Error: T must be a multiple of Delta'
    sys.exit()
params['M'] = params['T'] / params['Delta']

# Generate thetas
theta = np.zeros((params['N'], params['N'], params['L']))
method_name, method_params = params['theta_method']
if method_name == 'sparse_unique':
    for i in range(params['N']):
        if random() < method_params['p']:
            j = np.random.randint(0, params['N'])
            l = np.random.randint(0, params['L'])
            theta[i,j,l] = np.random.normal(0, method_params['scale'])
if method_name == 'cascade_2':
    for i in range(params['N']-1):
        theta[i,(i+1),1] = method_params['strength']*(method_params['decay']**i)
    theta[params['N']-1,0,0] = 0.5 * method_params['strength']
   
# Generate S (calling it "windows" in code)
print 'Generating window permutations'
windows = []
method_name, method_params = params['S_method']
if method_name in ['random_uniform', 'random_periodic']:
    if method_name == 'random_periodic':
        periods = params['M'] * np.random.normal(1, 1, params['N'])
        phases = np.random.uniform(0.0, 2.0*np.pi, params['N'])
    for k in range(params['M']):
        window = []
        if method_name == 'random_uniform':
            p = np.random.uniform(method_params['p_min'],method_params['p_max'])
            w_raw = np.random.binomial(1, p, (params['N'],params['Delta']))
        if method_name == 'random_periodic':
            w_raw = np.empty((params['N'], params['Delta']))
            for i in range(params['N']):
                p = (method_params['baseline'] +
                     method_params['scale'] * np.sin(periods[i]*k + phases[i]))
                w_raw[i,:] = np.random.binomial(1, p, (1, params['Delta']))
        w = np.array(w_raw, dtype='uint8')
        windows.append(window_permutations(w))
n_w = map(len, windows)

# Tabulate log-potential functions: h_1, h_2, ..., h_M
print 'Tabulating log-potential functions'
h = [np.empty(n_w[0])]
s_padded = np.zeros((params['N'],2*params['Delta']), dtype='uint8')
hits = np.zeros((params['N'],params['N'],params['L']), dtype='int32')
for w, s in enumerate(windows[0]):
    s_padded[:,params['Delta']:(2*params['Delta'])] = s
    hits[:,:,:] = 0
    for l in range(params['L']):
        tmin, tmax = params['Delta'] - (l+1), 2*params['Delta'] - (l+1)
        s_lagged = s_padded[:,tmin:tmax]
        hits[:,:,l] = np.tensordot(s_lagged, s, axes = (1,1))
    h[0][w] = np.sum(theta * hits)
for k in range(1, params['M']):
    h.append(np.empty((n_w[k-1], n_w[k])))
    for w_prev, s_prev in enumerate(windows[k-1]):
        s_padded[:,0:params['Delta']] = s_prev
        for w, s in enumerate(windows[k]):
            s_padded[:,params['Delta']:(2*params['Delta'])] = s
            hits[:,:,:] = 0
            for l in range(params['L']):
                tmin, tmax = params['Delta'] - (l+1), 2*params['Delta'] - (l+1)
                s_lagged = s_padded[:,tmin:tmax]
                hits[:,:,l] = np.tensordot(s_lagged, s, axes = (1,1))
            h[k][w_prev,w] = np.sum(theta * hits)

# Run the backward algorithm
print 'Running backward algorithm'
b = [np.zeros(n_w[params['M']-1])]            
for k in range(params['M']-1, 0, -1):
    b = [np.empty(n_w[k-1])] + b
    for w_prev in range(n_w[k-1]):
        for w in range(n_w[k]):
            b[0][w_prev] = np.logaddexp.reduce(h[k][w_prev,:] + b[1])
b = [None] + b

# Sample (binned) spike trains by sampling permutations of jitter windows
print 'Sampling'
x = np.empty((params['N'], params['T']), dtype='uint8')
w_samp = log_weighted_sample(h[0] + b[1])
x[:,0:params['Delta']] = windows[0][w_samp]
for k in range(1, params['M']):
    w_samp = log_weighted_sample(h[k][w_samp,:] + b[k+1])
    x[:,(k*params['Delta']):((k+1)*params['Delta'])] = windows[k][w_samp]

# Write sample to file
savemat('sample.mat', {'theta': theta, 'sample': np.array(x, dtype='float32')},
        oned_as = 'column')

# Output
print 'Parameters'
for param in params:
    print '%s: %s' % (param, str(params[param]))
print

print 'Theta'
print theta
print

print 'x'
print x
print

print 'log_kappa'
print np.logaddexp.reduce(h[0] + b[1])
