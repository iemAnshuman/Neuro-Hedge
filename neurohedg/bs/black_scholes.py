
"""
Black–Scholes pricing with numerically-stable d1/d2 and analytic call/put prices.
No external dependencies beyond numpy.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erf

# standard normal pdf and cdf (use erf for higher precision and speed)
SQRT_2PI = np.sqrt(2.0 * np.pi)

def _norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * np.asarray(x)**2) / SQRT_2PI

def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    # 0.5 * [1 + erf(x/sqrt(2))]
    return 0.5 * (1.0 + erf(np.asarray(x) / np.sqrt(2.0)))

def _d1(S, K, r, sigma, T, q=0.0):
    S, K, r, sigma, T, q = map(np.asarray, (S, K, r, sigma, T, q))
    # Guard small or zero times to expiry
    eps = 1e-12
    T = np.maximum(T, eps)
    vsqrt = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / np.maximum(vsqrt, eps)

def _d2(S, K, r, sigma, T, q=0.0):
    return _d1(S, K, r, sigma, T, q) - sigma * np.sqrt(np.maximum(T, 1e-12))

def bs_call_price(S, K, r, sigma, T, q=0.0):
    """European call price under Black–Scholes with continuous dividend yield q."""
    S, K, r, sigma, T, q = map(np.asarray, (S, K, r, sigma, T, q))
    d1 = _d1(S, K, r, sigma, T, q)
    d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-12))
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    return disc_q * S * _norm_cdf(d1) - disc_r * K * _norm_cdf(d2)

def bs_put_price(S, K, r, sigma, T, q=0.0):
    """European put price under Black–Scholes with continuous dividend yield q."""
    S, K, r, sigma, T, q = map(np.asarray, (S, K, r, sigma, T, q))
    d1 = _d1(S, K, r, sigma, T, q)
    d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-12))
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    return disc_r * K * _norm_cdf(-d2) - disc_q * S * _norm_cdf(-d1)

def put_call_parity_call_from_put(P, S, K, r, T, q=0.0):
    S, K, r, T, q, P = map(np.asarray, (S, K, r, T, q, P))
    return P + np.exp(-q*T)*S - np.exp(-r*T)*K

def put_call_parity_put_from_call(C, S, K, r, T, q=0.0):
    S, K, r, T, q, C = map(np.asarray, (S, K, r, T, q, C))
    return C - np.exp(-q*T)*S + np.exp(-r*T)*K
