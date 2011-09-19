#!/usr/bin/env python

# Trying to generate samples from the conditional model.

# NOTE: My convention of indexing arrays from 0 does not agree with
# the convention in Matt's notes. I'm also switching the definition of
# the lag in the more natural direction, as discussed in meeting.

# Daniel Klein, 9/14/2011

import sys
from itertools import permutations
import numpy as np
from random import random

# Parameters
params = {'N': 6,
          'T': 30,
          'L': 4,
          'Delta': 6,
          'theta_method': ('sparse_unique', {'p': 0.8, 'scale': 2.0}),
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

# Generate S (calling it "windows" in code)
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

# Define weighted sampling, with no optimization for repeated use
def log_weighted_sample(x, log_probs):
    log_probs = np.array(log_probs)
    log_probs_scaled = log_probs - np.max(log_probs)
    probs_unnorm = np.exp(log_probs_scaled)
    probs = probs_unnorm / np.sum(probs_unnorm)

    r = random()
    p_cum = 0.0
    for i, p in enumerate(probs):
        p_cum += p
        if r < p_cum: break
    return x[i]

# Sample (binned) spike trains by sampling permutations of jitter windows
x = np.zeros((params['N'], params['T']), dtype=int)
log_probs = []
for s in windows[0]:
    log_prob = 0.0
    for i in range(params['N']):
        for j in range(params['N']):
            for l in range(params['L']):
                this_theta = theta[i,j,l]
                hits = 0
                for t in range(params['Delta']):
                    t_lagged = t - (l + 1)
                    if t_lagged < 0: continue
                    hits += (s[i,t] * s[j,t_lagged])
                log_prob += this_theta * hits
    log_probs.append(log_prob)
x[:,0:params['Delta']] = log_weighted_sample(windows[0], log_probs)
for k in range(1, params['M']):
    log_probs = []
    t_min, t_max = params['Delta']*k, params['Delta']*(k+1)
    for s in windows[k]:
        log_prob = 0.0
        for i in range(params['N']):
            for j in range(params['N']):
                for l in range(params['L']):
                    this_theta = theta[i,j,l]
                    hits = 0
                    for t in range(t_min, t_max):
                        t_lagged = t - (l + 1)
                        if t_lagged < t_min:
                            hits += (s[i,t-t_min] * x[j,t_lagged])
                        else:
                            hits += (s[i,t-t_min] * s[j,t_lagged-t_min])
                    log_prob += this_theta * hits
        log_probs.append(log_prob)
    x[:,t_min:t_max] = log_weighted_sample(windows[k], log_probs)

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
