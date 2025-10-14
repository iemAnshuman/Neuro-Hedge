
# NeuroHedg · Phase 1

This starter includes:
- Black–Scholes pricing (call/put) with dividend yield `q`
- Greeks (Δ, Γ, vega, θ, ρ)
- Implied volatility solver (bracketed, robust)
- No-arbitrage checks and parity helpers
- Pytest unit tests with canonical numbers

## Quickstart
```bash
cd neurohedg_phase1
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac) or .venv\Scripts\activate on Windows
pip install -e .[dev]
pytest
```

## API sketch
```python
from neurohedg.bs import bs_call_price, bs_put_price, implied_vol, call_delta, vega
C = bs_call_price(S=100,K=100,r=0.05,sigma=0.2,T=1.0,q=0.0)
iv = implied_vol('c', C, 100,100,0.05,1.0,0.0)
```

This is the baseline for Phase 2 (error maps + hedge PnL).
