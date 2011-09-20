# Putting general-purpose functions here.
# Daniel Klein, 9/20/2011

import numpy as np
from random import random

# (Log-)weighted sampling, with no optimization for repeated use
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
