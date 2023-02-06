import numpy as np
from functools import wraps
from time import time

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


class binomial_tree:

    '''
    Binomial asset tree model with different options.

    We need to know several things:
    u, d, S0, T, N, r, K, option
    '''
    def __init__(self, params):

        self.__dict__.update(params)    # load model parameters
        self.dt = self.T/self.N         # Compute time step
        self.disc = np.exp(-self.r*self.dt)     # Compute discount rate
        self.p = (1/self.disc - self.d) / (self.u - self.d)  # Compute risk neutral probability
        self.delta0 = self.calculate_delta0()
        self.V0 = self.calculate_V0()
        self.X0 = self.calculate_V0()

    def calculate_Sn(self, n):
        # Compute potential price outcomes
        return self.S0 * self.d ** (np.arange(n, -1, -1)) * self.u ** (np.arange(0, n + 1, 1))

    def VN(self):
        # Compute asset prices at maturity T = N
        return np.maximum(self.calculate_Sn(self.N)-self.K, np.zeros(self.N+1))

    def calculate_V0(self):
        # Compute initial asset price through tree at time n
        C = self.VN()
        for i in np.arange(self.N, 0, -1):
          C = self.disc * (self.p * C[1:i+1] + (1-self.p) * C[0:i])

        return C[0]

    def calculate_Vn(self, n):
        # Compute initial asset price through tree at time n
        C = self.VN()
        for i in np.arange(self.N, n, -1):
            C = self.disc * (self.p * C[1:i + 1] + (1 - self.p) * C[0:i])

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
        return sum(self.disc * self.calculate_Vn(1) * np.array([1-self.p, self.p]))

    # def calculate_Xn(self, n):
    #     # Calculate wealth an maturation of european option
    #     xn = self.X0
    #     for i range(n):
    #         dn = self.calculate_deltan(i)
    #         sn = self.calculate_Sn(i)
    #         xn = self.d0 * self.calculate_Sn(1) + (1/self.disc) * (self.V0 - self.d0*self.S0)
    #
    #     return self.X1


if __name__ == "__main__":

    u = 1.1

    params = {'S0': 100,
                'r': 0.06,
                'u': u,
                'd': 1/u,
                'K': 100,
                'X0': 1,
                'N': 10,
                'T': 1}

    m = binomial_tree(params)
    # print(np.mean(m.calculate_X1()))
    # print(np.mean(m.calculate_Vn(1)))



