import numpy as np
from functools import wraps
from time import time
import cvxpy as cp
from scipy.special import binom

"""
Implementation of the Binomial Asset Tree model.
-----------------------------------------------

This is a pretty simple implementation of the BAT model.
It is broken into first a BinomialAssetTree class which computes all the underlying properties of the tree
This object is then inherited by different option classes which compute the option prices.

"""

class BinomialAssetTree:

    """
    This class implements the binomial asset tree model.

    Attributes:
        u (float): up factor
        S0 (float): initial asset price
        T (float): maturity
        N (int): number of time steps
        r (float): risk free rate
        K (float): strike price
    """

    def __init__(self, u, S0, K, T, N, r):
        self.u = u
        self.d = 1/u
        self.S0 = S0
        self.T = T
        self.N = N
        self.r = r
        self.K = K

        # Define some useful quantities
        self.dt = self.T/self.N                   # Time step
        self.disc = np.exp(-r*self.dt)  # Discount factor
        self.p = (self.disc - self.d)/(self.u - self.d) # Risk neutral probability

    def stock_prices(self, n=None):
        # Computes all possible stock prices at time step n
        if n is None:
            n = self.N

        return (
            self.S0
            * self.d ** (np.arange(n, -1, -1))
            * self.u ** (np.arange(0, n + 1, 1))
        )
        
    def risk_probs(self, n=None):
        '''
        Compute risk neutral probabilities of outcomes Sn
        Notes, we sume probabilities of symmetric events i.e P(HT) = P(TH) thus, multiply by 2
        '''
        if n is None:
            n = self.N

        risk_probs = (
            self.p ** (np.arange(n, -1, -1))
            * (1 - self.p) ** (np.arange(0, n + 1, 1))
            * binom(n, np.arange(n + 1))
        )

        return risk_probs

class EuropeanBAT(BinomialAssetTree):
    '''
    European option pricing using the Binomial Asset Tree model
    Inherits the BinomialAssetTree class
    '''

    def __init__(self, u, S0, K, T, N, r, option='call'):
        super().__init__(u, S0, K, T, N, r)
        self.option = option
        self.initial_price = self.price_option()
        self.initial_option_price = self.price_option()   # initial price of options
        self.initial_hedge = self.hedging_ratio()  # initial hedging ratio

    def payoff(self):
        # Define the payoff function
        if self.option == 'call':
            return np.maximum(self.stock_prices(self.N) - self.K, np.zeros(self.N + 1))
        else:
            return np.maximum(self.K - self.stock_prices(self.N), np.zeros(self.N + 1))

    def price_option(self, n=None):
        # price the option at time step n

        if n is None:
            # Default to find initial price
            n = 0

        # Initialise the option price vector
        Vn = self.payoff()
        for i in np.arange(self.N, n, -1):
            Vn = ((self.p) * Vn[1 : i + 1] + (1 - self.p) * Vn[0:i]) / self.disc

        return Vn

    def hedging_ratio(self, n=None):
        # Calculate hedging ratio
        if n is None:
            n = 0

        return np.diff(self.price_option(n+1)) / np.diff(self.stock_prices(n + 1))

class AmericanBAT(BinomialAssetTree):
    '''
    American option pricing using the Binomial Asset Tree model
    Inherits the BinomialAssetTree class
    '''

    def __init__(self, u, S0, K, T, N, r, option='put'):
        super().__init__(u, S0, K, T, N, r)
        self.option = option
        self.initial_price = self.price_option()
        self.initial_option_price = self.price_option()   # initial price of options
        self.initial_hedge = self.hedging_ratio()  # initial hedging ratio

    
    def payoff(self):
        # Compute potential price outcomes
        return np.maximum(self.K - self.stock_prices(self.N), 0)
    
    def price_option(self, n=None):
        # price the option at time step n
        if n is None:
            n = 0

        # Initialise the option price vector
        Vn = self.payoff()
        
        for i in np.arange(self.N, n, -1):
            # Compute Vn for each step
            Vn = ((self.p) * Vn[1 : i + 1] + (1 - self.p) * Vn[0:i]) / self.disc
            Vn = np.maximum(Vn, self.K - self.stock_prices(i-1))

        return Vn
    
    def hedging_ratio(self, n=None):
        # Calculate hedging ratio
        if n is None:
            n = 0

        return np.diff(self.price_option(n+1)) / np.diff(self.stock_prices(n + 1))
    
    def cash_flow(self, n=None):
        # Calculate cash flow back to investor at time step n
        if n==None:
            n = self.N - 1

        if n >= self.N:
            raise ValueError('AmericanBAT: n must be less than N for cash flow calculation')
        
        Vnp1 = self.price_option(n+1)

        return self.price_option(n) - (self.p * Vnp1[1:n+2] + (1 - self.p) * Vnp1[:n+1]) / self.disc
    
class CapitalAssetPricing(BinomialAssetTree):

    '''
    ########## NOT FINISHED ###########
    Implements the Capital Asset Pricing Model using the Binomial Asset Tree model
    Here we only use a log utility function
    Inherits the BinomialAssetTree class
    '''

    def __init__(self, u, S0, K, T, N, r, real_p, X0):
        '''
        Parameters
        ----------
        real_p : float  # real probabilities of an upward jump; does not distingusish between P(HT) and P(TH)
        X0 : float  # initial wealth
        '''
        super().__init__(u, S0, K, T, N, r)
        self.real_p = real_p    # real probabilities of an upward jump
        self.X0 = X0    # initial wealth

        # Compute risk neutral probabilities for all outcomes
        self.risk_ps = self.risk_probs()
        self.Z = self.risk_probs / self.real_p  # compute Randon Nikodym derivative
        self.zeta = self.Z / (1 + self.r) ** self.n # compute zeta
        self.pn_zeta = self.p * self.zeta # compute pn_zeta

    def solve(self):

        # Only implemented for log at the momen Generalise to arbitary convex function

        self.x = cp.Variable(self.N + 1)
        self.problem = cp.Problem(
            cp.Maximize(self.p @ cp.log(self.x)), [self.pn_zeta @ self.x == self.X0]
        )
        self.problem.solve()





if __name__ == "__main__":

    N = 3
    u = 2
    r = -np.log(1.25)
    K = 5
    S0 = 4
    T = N
    option = "put"


    tree = BinomialAssetTree(u, S0, K, T, N, r)
    eurotree = EuropeanBAT(u, S0, K, T, N, r, option=option)
    ustree = AmericanBAT(u, S0, K, T, N, r)

    print(eurotree.hedging_ratio())
    print(ustree.cash_flow(2))


