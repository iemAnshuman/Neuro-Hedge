import numpy as np
import math
from scipy.stats import norm

class BlackScholes:
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        if T <= 0:
            return np.inf if S > K else -np.inf
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma, d1_val):
        if T <= 0:
            return np.inf
        return d1_val - sigma * math.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0:
            return max(0.0, S - K)
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(S, K, T, r, sigma, d1_val)
        
        return S * norm.cdf(d1_val) - K * math.exp(-r * T) * norm.cdf(d2_val)

    @staticmethod
    def call_delta(S, K, T, r, sigma):
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1_val)

class EuropeanCallOption:
    
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.name = f"Call_K{K}_T{T}"

    def payoff(self, S_T):
        return max(0.0, S_T - self.K)

class BlackScholesBaseline:
    
    def __init__(self, K, T, r, sigma):
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def get_action(self, t, S_t):
        time_to_expiry = self.T - t
        delta = BlackScholes.call_delta(S_t, self.K, time_to_expiry, self.r, self.sigma)
        return delta

if __name__ == '__main__':
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    price = BlackScholes.call_price(S, K, T, r, sigma)
    delta = BlackScholes.call_delta(S, K, T, r, sigma)
    
    print(f"--- Black-Scholes Test ---")
    print(f"Stock Price: {S}, Strike: {K}, Time: {T}, Vol: {sigma}")
    print(f"Option Price: {price:.4f}")
    print(f"Option Delta: {delta:.4f}")
    
    option = EuropeanCallOption(K, T)
    print(f"\nOption Payoff at S=110: {option.payoff(110.0)}")
    print(f"Option Payoff at S=90: {option.payoff(90.0)}")

    baseline_bot = BlackScholesBaseline(K, T, r, sigma)
    action_t0 = baseline_bot.get_action(t=0.0, S_t=100.0)
    action_t_half = baseline_bot.get_action(t=0.5, S_t=110.0)

    print(f"\n--- Baseline Bot Test ---")
    print(f"Hedge (Delta) at t=0, S=100: {action_t0:.4f}")
    print(f"Hedge (Delta) at t=0.5, S=110: {action_t_half:.4f}")