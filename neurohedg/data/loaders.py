
"""
Lightweight data loaders; yfinance optional for quick prototyping.
"""
from __future__ import annotations
import datetime as dt

def fetch_ohlc_yf(ticker: str, start: str = "2023-01-01", end: str | None = None):
    """
    Returns a pandas DataFrame if yfinance/pandas are installed.
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not installed; run `pip install yfinance pandas`") from e
    end = end or dt.date.today().isoformat()
    return yf.download(ticker, start=start, end=end, progress=False)
