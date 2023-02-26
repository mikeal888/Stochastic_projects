import numpy as np
from functools import wraps
from time import time
import cvxpy as cp
from scipy.special import binom


def timing(f):
    # Create timing wrapper
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


class BinomialTreeEuropean:

    """
    Binomial asset tree model with different options.

    We need to know several things:
    u, d, S0, T, N, r, K, option
    """
    def __init__(self, params):

        self.__dict__.update(params)  # load model parameters
        self.dt = self.T / self.N  # Compute time step
        self.disc = 1 + self.r  # Compute discount rate
        self.risk_p = (self.disc - self.d) / (
            self.u - self.d
        )  # Compute risk neutral probability
        self.delta0 = self.calculate_delta0()
        self.V0 = self.calculate_v0()
        self.X0 = self.calculate_x0()

    def calculate_sn(self, n):
        # Compute potential price outcomes
        return (
            self.S0
            * self.d ** (np.arange(n, -1, -1))
            * self.u ** (np.arange(0, n + 1, 1))
        )

    def vn(self):
        # Compute asset prices at maturity T = N
        if self.option == "put":
            return np.maximum(self.K - self.calculate_sn(self.N), np.zeros(self.N + 1))
        else:
            return np.maximum(self.calculate_sn(self.N) - self.K, np.zeros(self.N + 1))

    def calculate_v0(self):
        # Compute initial asset price through tree at time n
        C = self.vn()
        for i in np.arange(self.N, 0, -1):
            # C[1:i+1] corresponds to all tails at the end
            C = ((self.risk_p) * C[1 : i + 1] + (1 - self.risk_p) * C[0:i]) / self.disc

        return C[0]

    def calculate_vn(self, n):
        # Compute initial asset price through tree at time n
        C = self.vn()
        for i in np.arange(self.N, n, -1):
            C = ((self.risk_p) * C[1 : i + 1] + (1 - self.risk_p) * C[0:i]) / self.disc

        return C

    def calculate_delta0(self):
        # Compute risk neutal share holding
        self.d0 = np.diff(self.calculate_vn(1))[0] / ((self.u - self.d) * self.S0)
        return self.d0

    def calculate_deltan(self, n):
        # Calculate
        return np.diff(self.calculate_vn(n + 1)) / np.diff(self.calculate_sn(n + 1))

    def calculate_x0(self):
        # Calculate X0
        return sum(
            (1 / self.disc)
            * self.calculate_vn(1)
            * np.array([1 - self.risk_p, self.risk_p])
        )


class CapitalAssetPricing():

    # Here we implement the capital asset pricing model using cvx. Here we will need a specified n,
    # and real probabilities

    def __init__(self, params):
        self.__dict__.update(params)  # load model parameters

        self.disc = 1 + self.r  # Compute discount rate

        # Compute risk probabilities but scaled by binom for symmetry
        self.risk_p = (self.disc - self.d) / (self.u - self.d)
        self.risk_probs = self.calculate_risk_probs()

        self.Z = self.risk_probs / self.p  # compute Randon Nikodym derivative
        self.zeta = self.Z / (1 + self.r) ** self.n
        self.pn_zeta = self.p * self.zeta

    def calculate_sn(self):
        # Compute potential price outcomes
        return (
            self.S0
            * self.d ** (np.arange(self.n, -1, -1))
            * self.u ** (np.arange(0, self.n + 1, 1))
        )

    def calculate_risk_probs(self):
        # Compute risk neutral probabilities
        risk_probs = (
            self.risk_p ** (np.arange(self.n, -1, -1))
            * (1 - self.risk_p) ** (np.arange(0, self.n + 1, 1))
            * binom(self.n, np.arange(self.n + 1))
        )

        return risk_probs

    def solve(self):

        # Only implemented for log at the moment
        # Generalise to arbitary convex function

        self.x = cp.Variable(self.n + 1)
        self.problem = cp.Problem(
            cp.Maximize(self.p @ cp.log(self.x)), [self.pn_zeta @ self.x == self.X0]
        )
        self.problem.solve()


if __name__ == "__main__":

    n = 2
    u = 2
    r = 0.25

    p = np.array([1 / 9, 4 / 9, 4 / 9])
    X0 = 4

    params = {"r": r, "u": u, "d": 1 / u, "X0": X0, "p": p, "n": n}

    model = CapitalAssetPricing(params)

    model.solve()

    print(model.x.value)

    print(model.problem.status)


