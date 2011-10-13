# Putting general-purpose functions here.
# Daniel Klein, 9/20/2011

import numpy as np
from random import random
from itertools import permutations
import hashlib
from os import system

def logaddexp(x):
    return np.logaddexp.reduce(x)

def fast_average(x, weights):
    weights = np.array(weights, copy = False, ndmin = x.ndim).swapaxes(-1, 0)
    return np.multiply(x, weights).sum(axis = 0) / weights.sum(axis = 0)

def unlog(log_x):
    log_x = np.array(log_x)
    log_x_scaled = log_x - np.max(log_x)
    x_unnorm = np.exp(log_x_scaled)
    x = x_unnorm / np.sum(x_unnorm)
    return x

# (Log-)weighted sampling, with no optimization for repeated use
def log_weighted_sample(log_probs):
    probs = unlog(log_probs)

    r = random()
    p_cum = 0.0
    for i, p in enumerate(probs):
        p_cum += p
        if r < p_cum: break
    return i

def hash_array(x):
    return hashlib.sha1(x.view(np.uint8)).hexdigest()

def window_permutations(w):
    n = w.shape[1]

    perms = []
    w_seen = set()
    for perm in permutations(range(n)):
        w_perm = w[:,np.array(perm)]
        w_hash = hash_array(w_perm)
        if w_hash in w_seen: continue
        w_seen.add(w_hash)
        perms.append(w_perm)

    return perms

def permute(w):
    return next_permutation(sorted(w))

# Credit: http://blog.bjrn.se/2008/04/lexicographic-permutations-using.html
# Removed unnecessary code as appropriate.
def next_permutation(seq):
    """Like C++ std::next_permutation() but implemented as
    generator. Yields copies of seq."""

    def reverse(seq, start, end):
        # seq = seq[:start] + reversed(seq[start:end]) + \
        #       seq[end:]
        end -= 1
        if end <= start:
            return
        while True:
            seq[start], seq[end] = seq[end], seq[start]
            if start == end or start+1 == end:
                return
            start += 1
            end -= 1
    
    first = 0
    last = len(seq)
    seq = seq[:]

    yield seq
    
    if last == 1:
        raise StopIteration

    while True:
        next = last - 1

        while True:
            # Step 1.
            next1 = next
            next -= 1

            if seq[next] < seq[next1]:
                # Step 2.
                mid = last - 1
                while not seq[next] < seq[mid]:
                    mid -= 1
                seq[next], seq[mid] = seq[mid], seq[next]
                
                # Step 3.
                reverse(seq, next1, last)

                yield seq[:]
                break
            if next == first:
                raise StopIteration
    raise StopIteration

def theta_viz(theta, threshold = 0.01):
    outfile = open('theta_viz.dot', 'w')
    outfile.write('digraph G {\n')
    theta_n, theta_l = theta.shape[0], theta.shape[2]
    for i in range(theta_n):
        for j in range(theta_n):
            for l in range(theta_l):
                if abs(theta[i,j,l]) < threshold: continue
                outfile.write('%d -> %d\n' % (i, j))
    outfile.write('}\n')
    outfile.close()
    system('neato -Tpng theta_viz.dot -o theta_viz.png')
