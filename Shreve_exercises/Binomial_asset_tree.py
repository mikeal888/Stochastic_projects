import numpy as np

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec'% (f.__name__, args, kw, te-ts))
        return result
    return wrap


# Initialise parameters
S0 = 100
K = 100
T = 1
r = 0.06
N = 3
u = 1.1
d = 1/u
opttype = 'C'   # Could be P

# Define Binomial tree slow
@timing
def binomial_tree_slow(K, T, S0, r, N, u, d, opttype):
    # precompute constants
    dt = T/N
    q = (np.exp(r*dt) - d) / (u - d)    # risk neutral probability
    disc = np.exp(-r*dt)    # discount rates

    # initialise assest prices at maturity T = N
    S = np.zeros(N+1)
    S[0] = S0*d**N
    for j in range(1, N+1):
        S[j] = S[j-1]*u/d   # strange way of calculating binomial


    # initialise option values at maturity
    C = np.zeros(N+1)
    for j in range(0, N+1):
        C[j] = max(0, S[j]-K)

    # Step backwards through tree
    for i in np.arange(N, 0, -1):
        for j in range(0, i):
            C[j] = disc * (q*C[j+1] + (1-q)*C[j])

    return C[0]

# Define fast method
# Define Binomial tree slow
@timing
def binomial_tree_fast(K, T, S0, r, N, u, d, opttype):
    # precompute constants
    dt = T/N
    q = (np.exp(r*dt) - d) / (u - d)    # risk neutral probability
    disc = np.exp(-r*dt)    # discount rates

    # initialise assest prices at maturity T = N
    C = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N+1, 1))

    # initialise option values at maturity
    C = np.maximum(C-K, np.zeros(N+1))

    return C

    # Step backwards through tree
    for i in np.arange(N, 0, -1):
        C = disc * (q * C[1:i+1] + (1-q) * C[0:i])

    return C



# print(binomial_tree_slow(K, T, S0, r, N, u, d, opttype))

print(binomial_tree_fast(K, T, S0, r, N, u, d, opttype))