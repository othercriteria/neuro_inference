#!/usr/bin/env python

# Trying to generate samples from the conditional model.

# NOTE: My convention of indexing arrays from 0 does not agree with
# the convention in Matt's notes.

# Daniel Klein, 9/14/2011

import sys
from itertools import permutations
import numpy as np
from random import random

from utility import log_weighted_sample

# Parameters
params = {'N': 8,
          'T': 36,
          'L': 4,
          'Delta': 6,
          # 'theta_method': ('sparse_unique', {'p': 0.8, 'scale': 2.0}),
          'theta_method': ('cascade_2', {'strength': 8.0, 'decay': 0.8}),
          'S_method': ('random_uniform', {'p_min': 0.05, 'p_max': 0.2})}
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
   
# Generate S (calling it "windows" in code)
print 'Generating window permutations'
windows = []
method_name, method_params = params['S_method']
if method_name == 'random_uniform':
    for k in range(params['M']):
        window = []
        p = np.random.uniform(method_params['p_min'], method_params['p_max'])
        w = np.random.binomial(1, p, (params['N'], params['Delta']))
        w_seen = set()
        for perm in permutations(range(params['Delta'])):
            w_perm = w[:,np.array(perm)]
            w_str = np.array_str(w_perm)
            if w_str in w_seen:
                continue
            w_seen.add(w_str)
            window.append(w_perm)
        windows.append(window)
n_w = map(len, windows)

# Tabulate log-potential functions: h_1, h_2, ..., h_M (this has the
# feel of the forward algorithm?)
print 'Tabulating log-potential functions'
h = [np.empty(n_w[0])]
s_padded = np.zeros((params['N'],2*params['Delta']))
hits = np.zeros((params['N'],params['N'],params['L']))
for w, s in enumerate(windows[0]):
    s_padded[:,params['Delta']:(2*params['Delta'])] = s
    hits[:,:,:] = 0
    for l in range(params['L']):
        t_min, t_max = params['Delta'] - (l+1), 2*params['Delta'] - (l+1)
        s_lagged = s_padded[:,t_min:t_max]
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
                t_min, t_max = params['Delta'] - (l+1), 2*params['Delta'] - (l+1)
                s_lagged = s_padded[:,t_min:t_max]
                hits[:,:,l] = np.tensordot(s_lagged, s, axes = (1,1))
            h[k][w_prev,w] = np.sum(theta * hits)

# Run the backward algorithm
print 'Running backward algorithm'
b = [np.zeros(n_w[params['M']-1])]            
for k in range(params['M']-1, 0, -1):
    b = [np.empty(n_w[k-1])] + b
    for w_prev in range(n_w[k-1]):
        for w in range(n_w[k]):
            b[0][w_prev] = np.log(np.sum(np.exp(h[k][w_prev,:] + b[1])))
b = [None] + b

# Sample (binned) spike trains by sampling permutations of jitter windows
print 'Sampling'
x = np.zeros((params['N'], params['T']), dtype=int)
w_samp = log_weighted_sample(h[0] + b[1])
x[:,0:params['Delta']] = windows[0][w_samp]
for k in range(1, params['M']):
    w_samp = log_weighted_sample(h[k][w_samp,:] + b[k+1])
    x[:,(k*params['Delta']):((k+1)*params['Delta'])] = windows[k][w_samp]

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
print np.log(np.sum(np.exp(h[0] + b[1])))
