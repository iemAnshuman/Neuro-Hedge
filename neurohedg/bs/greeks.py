
"""
Analytic Greeks for European calls/puts in Black–Scholes.
Conventions: returns per underlying, vega per 1.0 volatility (not 1%).
"""
from __future__ import annotations
import numpy as np
from .black_scholes import _d1, _d2, _norm_pdf, _norm_cdf

def call_delta(S, K, r, sigma, T, q=0.0):
    d1 = _d1(S, K, r, sigma, T, q)
    return np.exp(-q * T) * _norm_cdf(d1)

def put_delta(S, K, r, sigma, T, q=0.0):
    d1 = _d1(S, K, r, sigma, T, q)
    return np.exp(-q * T) * (_norm_cdf(d1) - 1.0)

def gamma(S, K, r, sigma, T, q=0.0):
    d1 = _d1(S, K, r, sigma, T, q)
    return (np.exp(-q * T) * _norm_pdf(d1)) / (S * sigma * np.sqrt(np.maximum(T, 1e-12)))

def vega(S, K, r, sigma, T, q=0.0):
    d1 = _d1(S, K, r, sigma, T, q)
    return np.exp(-q * T) * S * _norm_pdf(d1) * np.sqrt(np.maximum(T, 1e-12))

def call_theta(S, K, r, sigma, T, q=0.0):
    d1 = _d1(S, K, r, sigma, T, q); d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-12))
    term1 = - (np.exp(-q*T) * S * _norm_pdf(d1) * sigma) / (2.0 * np.sqrt(np.maximum(T, 1e-12)))
    term2 = - r * K * np.exp(-r*T) * _norm_cdf(d2)
    term3 = + q * S * np.exp(-q*T) * _norm_cdf(d1)
    return term1 + term2 + term3

def put_theta(S, K, r, sigma, T, q=0.0):
    d1 = _d1(S, K, r, sigma, T, q); d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-12))
    term1 = - (np.exp(-q*T) * S * _norm_pdf(d1) * sigma) / (2.0 * np.sqrt(np.maximum(T, 1e-12)))
    term2 = + r * K * np.exp(-r*T) * _norm_cdf(-d2)
    term3 = - q * S * np.exp(-q*T) * _norm_cdf(-d1)
    return term1 + term2 + term3

def call_rho(S, K, r, sigma, T, q=0.0):
    d1 = _d1(S, K, r, sigma, T, q); d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-12))
    return K * T * np.exp(-r*T) * _norm_cdf(d2)

def put_rho(S, K, r, sigma, T, q=0.0):
    d1 = _d1(S, K, r, sigma, T, q); d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-12))
    return -K * T * np.exp(-r*T) * _norm_cdf(-d2)
