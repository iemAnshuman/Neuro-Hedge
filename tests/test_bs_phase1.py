
import numpy as np
from neurohedg.bs.black_scholes import bs_call_price, bs_put_price
from neurohedg.bs.greeks import call_delta, put_delta, gamma, vega, call_theta, put_theta, call_rho, put_rho
from neurohedg.bs.implied_vol import implied_vol
from neurohedg.utils.noarb_checks import check_put_call_parity

# Canonical test case: S=100,K=100,r=5%,T=1, sigma=20%, q=0
S=100.0; K=100.0; r=0.05; T=1.0; sigma=0.2; q=0.0

def test_prices_match_known():
    C = bs_call_price(S,K,r,sigma,T,q)
    P = bs_put_price(S,K,r,sigma,T,q)
    # Known reference (rounded): 10.4506, 5.5735
    assert abs(C - 10.4506) < 1e-3
    assert abs(P - 5.5735) < 1e-3

def test_put_call_parity():
    C = bs_call_price(S,K,r,sigma,T,q)
    P = bs_put_price(S,K,r,sigma,T,q)
    ok, diff = check_put_call_parity(C,P,S,K,r,T,q, tol=1e-8)
    assert ok, f"Parity violated by {diff.max()}"

def test_implied_vol_recovers_sigma():
    C = bs_call_price(S,K,r,sigma,T,q)
    iv = implied_vol('c', C, S,K,r,T,q)
    assert abs(iv - sigma) < 2e-4

def test_greeks_shape_and_signs():
    Cdelta = call_delta(S,K,r,sigma,T,q)
    Pdelta = put_delta(S,K,r,sigma,T,q)
    g = gamma(S,K,r,sigma,T,q)
    v = vega(S,K,r,sigma,T,q)
    Ct = call_theta(S,K,r,sigma,T,q)
    Pt = put_theta(S,K,r,sigma,T,q)
    Cr = call_rho(S,K,r,sigma,T,q)
    Pr = put_rho(S,K,r,sigma,T,q)

    assert 0.0 < Cdelta < 1.0
    assert -1.0 < Pdelta < 0.0
    assert g > 0.0
    assert v > 0.0
    # Typically call theta negative for non-dividend assets at moderate params
    assert Ct < 0.0
    # Put theta can be negative or less negative; allow either sign but finite
    assert np.isfinite(Pt)
    assert Cr > 0.0 and Pr < 0.0
