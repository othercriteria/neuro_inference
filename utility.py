# Putting general-purpose functions here.
# Daniel Klein, 9/20/2011

import numpy as np
from random import random
from itertools import permutations
import hashlib

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



