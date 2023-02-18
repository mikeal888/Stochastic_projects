import numpy as np
from functools import wraps
from time import time
from scipy.special import binom

def timing(f):
    # Create timing wrapper
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec'% (f.__name__, args, kw, te-ts))
        return result
    return wrap

class binomial_tree_european:

    '''
    Binomial asset tree model with different options.

    We need to know several things:
    u, d, S0, T, N, r, K, option
    '''
    def __init__(self, params):

        self.__dict__.update(params)    # load model parameters
        self.dt = self.T/self.N         # Compute time step
        self.disc = (1+self.r)           # Compute discount rate
        self.risk_p = (self.disc - self.d) / (self.u - self.d)  # Compute risk neutral probability
        self.delta0 = self.calculate_delta0()
        self.V0 = self.calculate_V0()
        self.X0 = self.calculate_X0()

    def calculate_Sn(self, n):
        # Compute potential price outcomes
        return self.S0 * self.d ** (np.arange(n, -1, -1)) * self.u ** (np.arange(0, n + 1, 1))

    def VN(self):
        # Compute asset prices at maturity T = N
        if self.option == 'put':
            return np.maximum(self.K - self.calculate_Sn(self.N), np.zeros(self.N+1))
        else:
             return np.maximum(self.calculate_Sn(self.N)-self.K, np.zeros(self.N+1))

    def calculate_V0(self):
        # Compute initial asset price through tree at time n
        C = self.VN()
        for i in np.arange(self.N, 0, -1):
            # C[1:i+1] corresponds to all tails at the end 
            C =  ((self.risk_p) * C[1:i+1] + (1-self.risk_p) * C[0:i])/self.disc

        return C[0]

    def calculate_Vn(self, n):
        # Compute initial asset price through tree at time n
        C = self.VN()
        for i in np.arange(self.N, n, -1):
            C = ((self.risk_p) * C[1:i + 1] + (1-self.risk_p) * C[0:i])/self.disc

        return C

    def calculate_delta0(self):
        # Compute risk neutal share holding
        self.d0 = np.diff(self.calculate_Vn(1))[0]/((self.u-self.d)*self.S0)
        return self.d0

    def calculate_deltan(self, n):
        # Calculate
        return np.diff(self.calculate_Vn(n+1))/np.diff(self.calculate_Sn(n+1))

    def calculate_X0(self):
        # Calculate X0
        return sum((1/self.disc) * self.calculate_Vn(1) * np.array([1-self.risk_p, self.risk_p]))