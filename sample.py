#!/usr/bin/env python

# Trying to generate samples from the conditional model.

# NOTE: My convention of indexing arrays from 0 does not agree with
# the convention in Matt's notes

# Daniel Klein, 9/14/2011

import sys
import itertools
import numpy as np
from random import random

# Parameters
params = {'N': 5,
          'T': 30,
          'L': 4,
          'Delta': 6,
          'theta_method': ('sparse_unique', {'p': 0.8, 'scale': 2.0})}
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

# Sample (binned) spike trains
x = np.zeros((params['N'], params['T']), dtype=int)

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
