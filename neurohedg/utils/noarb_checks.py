
"""
Basic sanity/no-arbitrage checks and helpers.
"""
from __future__ import annotations
import numpy as np

def check_put_call_parity(call, put, S, K, r, T, q=0.0, tol=1e-6):
    call, put, S, K, r, T, q = map(np.asarray, (call, put, S, K, r, T, q))
    lhs = call - put
    rhs = np.exp(-q*T)*S - np.exp(-r*T)*K
    diff = np.abs(lhs - rhs)
    return np.all(diff <= tol), diff

def monotonicity_in_K(call_prices, strikes):
    """
    Call prices should be non-increasing in K (for fixed S,r,q,T).
    Returns True if monotone (within numerical jitter), else False.
    """
    call_prices = np.asarray(call_prices); strikes = np.asarray(strikes)
    diffs = np.diff(call_prices)
    return bool(np.all(diffs <= 1e-8)), diffs
