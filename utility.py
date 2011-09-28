# Putting general-purpose functions here.
# Daniel Klein, 9/20/2011

import numpy as np
from random import random
from itertools import permutations
import hashlib

# (Log-)weighted sampling, with no optimization for repeated use
def log_weighted_sample(log_probs):
    log_probs = np.array(log_probs)
    log_probs_scaled = log_probs - np.max(log_probs)
    probs_unnorm = np.exp(log_probs_scaled)
    probs = probs_unnorm / np.sum(probs_unnorm)

    r = random()
    p_cum = 0.0
    for i, p in enumerate(probs):
        p_cum += p
        if r < p_cum: break
    return i

def window_permutations(w):
    n = w.shape[1]

    perms = []
    w_seen = set()
    for perm in permutations(range(n)):
        w_perm = w[:,np.array(perm)]
        w_hash = hashlib.sha1(w_perm.view(np.uint8)).hexdigest()
        if w_hash in w_seen: continue
        w_seen.add(w_hash)
        perms.append(w_perm)

    return perms



