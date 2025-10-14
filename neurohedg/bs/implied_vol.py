
"""
Brent-style bracketed solver for implied volatility with safe fallbacks.
We solve for sigma in [1e-6, 5.0] by matching BS price to target.
"""
from __future__ import annotations
import numpy as np
from .black_scholes import bs_call_price, bs_put_price

def _price_fn(option_type: str):
    if option_type.lower() in ("c", "call"):
        return bs_call_price
    elif option_type.lower() in ("p", "put"):
        return bs_put_price
    else:
        raise ValueError("option_type must be 'call'/'c' or 'put'/'p'")

def implied_vol(option_type: str, market_price, S, K, r, T, q=0.0, tol=1e-8, max_iter=100):
    price_model = _price_fn(option_type)
    market_price = np.asarray(market_price, dtype=float)
    S = np.asarray(S, dtype=float); K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float); T = np.asarray(T, dtype=float); q = np.asarray(q, dtype=float)
    # broadcast to common shape
    out_shape = np.broadcast_shapes(market_price.shape, S.shape, K.shape, r.shape, T.shape, q.shape)
    m = np.broadcast_to(market_price, out_shape).astype(float)
    S = np.broadcast_to(S, out_shape); K = np.broadcast_to(K, out_shape)
    r = np.broadcast_to(r, out_shape); T = np.broadcast_to(T, out_shape); q = np.broadcast_to(q, out_shape)

    # Bracket
    lo = np.full(out_shape, 1e-6); hi = np.full(out_shape, 5.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = price_model(S, K, r, mid, T, q)
        too_low = p_mid < m
        lo = np.where(too_low, mid, lo)
        hi = np.where(~too_low, mid, hi)
        if np.all(np.max(hi - lo) < tol):
            break
    return 0.5 * (lo + hi)
