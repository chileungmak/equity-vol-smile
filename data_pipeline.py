"""Data Ingestion and Cleaning Pipeline for Options Chains.

Ingests intraday market quotes via yfinance, sanitizes raw order book noise,
applies microstructure filters (volume, bid-ask spreads, arbitrage bounds),
and computes contract metrics (mid-prices, moneyness, time-to-expiration).
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_metadata(ticker_symbol: str) -> Dict[str, Any]:
    """Fetch underlying spot price and available expiration dates for a ticker.

    Cached for 300 seconds to preserve UI responsiveness while tracking intraday updates.

    Parameters
    ----------
    ticker_symbol : str
        Equity ticker symbol (e.g., 'SPY', 'AAPL').

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'ticker': yf.Ticker object
        - 'spot_price': float
        - 'expirations': List[str] (sorted list of available expiry dates 'YYYY-MM-DD')
        - 'currency': str

    Raises
    ------
    ValueError
        If the ticker symbol is invalid or contains no options data.
    """
    clean_symbol = ticker_symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("Ticker symbol cannot be empty.")

    try:
        ticker = yf.Ticker(clean_symbol)
        expirations: Tuple[str, ...] = ticker.options
    except Exception as exc:
        raise ValueError(f"Failed to query options for ticker '{clean_symbol}': {exc}") from exc

    if not expirations:
        raise ValueError(
            f"No options chain data available for '{clean_symbol}'. "
            "Verify the ticker symbol or market trading status."
        )

    # Robust spot price resolution across fast_info, regularMarketPrice, and 1-day history
    spot_price: Optional[float] = None

    # Priority 1: fast_info last_price or previous_close
    try:
        if hasattr(ticker, "fast_info"):
            price = getattr(ticker.fast_info, "last_price", None)
            if price is None or np.isnan(price) or price <= 0:
                price = getattr(ticker.fast_info, "previous_close", None)
            if price is not None and not np.isnan(price) and price > 0:
                spot_price = float(price)
    except Exception:
        pass

    # Priority 2: info regularMarketPrice
    if spot_price is None:
        try:
            info = ticker.info
            price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            if price is not None and not np.isnan(price) and price > 0:
                spot_price = float(price)
        except Exception:
            pass

    # Priority 3: 1-day intraday/daily history
    if spot_price is None:
        try:
            hist = ticker.history(period="5d")
            if not hist.empty and "Close" in hist.columns:
                spot_price = float(hist["Close"].dropna().iloc[-1])
        except Exception:
            pass

    if spot_price is None or spot_price <= 0:
        raise ValueError(f"Unable to retrieve a valid spot price for '{clean_symbol}'.")

    # Currency extraction
    currency = "USD"
    try:
        if hasattr(ticker, "fast_info") and hasattr(ticker.fast_info, "currency"):
            currency = ticker.fast_info.currency or "USD"
    except Exception:
        pass

    return {
        "ticker_symbol": clean_symbol,
        "spot_price": spot_price,
        "expirations": list(expirations),
        "currency": currency,
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_chain_raw(ticker_symbol: str, expiry_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve raw call and put options chains for a specified expiration.

    Parameters
    ----------
    ticker_symbol : str
        Ticker symbol.
    expiry_date : str
        Expiration date string in 'YYYY-MM-DD' format.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing raw (calls_df, puts_df).

    Raises
    ------
    ValueError
        If the option chain cannot be retrieved.
    """
    clean_symbol = ticker_symbol.strip().upper()
    try:
        ticker = yf.Ticker(clean_symbol)
        chain = ticker.option_chain(expiry_date)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch option chain for {clean_symbol} on {expiry_date}: {exc}"
        ) from exc

    return calls, puts


def calculate_time_to_expiration(expiry_date_str: str) -> Tuple[float, int]:
    """Calculate annualized time to expiration (T) and calendar days to expiration (DTE).

    Assumes market expiration at 16:00 (4:00 PM) Eastern Time.
    Enforces a strict positive floor for 0DTE contracts to avoid division by zero.

    Parameters
    ----------
    expiry_date_str : str
        Expiration date in 'YYYY-MM-DD' format.

    Returns
    -------
    Tuple[float, int]
        (T_annualized, days_to_expiration).
    """
    expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    today = date.today()
    dte = (expiry_date - today).days

    if dte <= 0:
        # 0DTE option: allocate remaining fraction of trading day (~4 hours or 0.5/365.0 minimum)
        T = max(1.0 / (365.0 * 24.0), 0.5 / 365.0)
        return float(T), 0

    T = dte / 365.0
    return float(T), int(dte)


def clean_options_data(
    raw_df: pd.DataFrame,
    spot_price: float,
    option_type: str = "call",
    min_volume: int = 1,
    min_bid: float = 0.05,
    max_spread_pct: float = 0.60,
    min_open_interest: int = 0,
) -> pd.DataFrame:
    """Filter out noisy, illiquid, or arbitrage-violating option quotes.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw calls or puts DataFrame from yfinance.
    spot_price : float
        Current underlying spot price.
    option_type : str, default 'call'
        'call' or 'put'.
    min_volume : int, default 1
        Filter strikes with volume strictly below this threshold.
    min_bid : float, default 0.05
        Filter out strikes with zero or negligible bid prices (market maker absence).
    max_spread_pct : float, default 0.60
        Filter out strikes with bid-ask spreads exceeding this fraction of mid-price.
    min_open_interest : int, default 0
        Filter contracts with open interest below threshold.

    Returns
    -------
    pd.DataFrame
        Sanitized, validated DataFrame ready for quantitative model inversion.
    """
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # Fill NaN values for numerical columns
    df["bid"] = pd.to_numeric(df.get("bid", 0.0), errors="coerce").fillna(0.0)
    df["ask"] = pd.to_numeric(df.get("ask", 0.0), errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df["openInterest"] = pd.to_numeric(df.get("openInterest", 0), errors="coerce").fillna(0)
    df["strike"] = pd.to_numeric(df.get("strike", 0.0), errors="coerce").fillna(0.0)

    # 1. Calculate mid-price and spread metrics
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
    df["bid_ask_spread"] = df["ask"] - df["bid"]

    # Avoid division by zero when calculating spread percentage
    safe_mid = np.where(df["mid_price"] > 0, df["mid_price"], np.nan)
    df["spread_pct"] = df["bid_ask_spread"] / safe_mid

    # 2. Liquidity & Order Book Filters
    # - Bid must be positive (meaning there is an active market maker)
    # - Ask must strictly exceed bid (eliminates crossed or locked quotes)
    # - Volume >= min_volume
    # - Open Interest >= min_open_interest
    mask_liquidity = (
        (df["bid"] >= min_bid)
        & (df["ask"] > df["bid"])
        & (df["volume"] >= min_volume)
        & (df["openInterest"] >= min_open_interest)
    )

    # 3. Spread Noise Filter
    mask_spread = (df["spread_pct"] <= max_spread_pct) & (df["spread_pct"] >= 0.0)

    # 4. No-Arbitrage Boundary Sanity Filter
    # An option mid price should not trade below its immediate intrinsic payoff.
    # Stale quotes after hours often violate intrinsic value, causing solver breakdown.
    if option_type == "call":
        intrinsic = np.maximum(0.0, spot_price - df["strike"])
        mask_intrinsic = (df["mid_price"] > intrinsic) & (df["mid_price"] < spot_price)
    else:
        intrinsic = np.maximum(0.0, df["strike"] - spot_price)
        mask_intrinsic = (df["mid_price"] > intrinsic) & (df["mid_price"] < df["strike"])

    combined_mask = mask_liquidity & mask_spread & mask_intrinsic
    clean_df = df[combined_mask].copy()

    # Drop duplicate strikes if any, keeping highest volume entry
    clean_df.sort_values(by=["strike", "volume"], ascending=[True, False], inplace=True)
    clean_df.drop_duplicates(subset=["strike"], keep="first", inplace=True)
    clean_df.reset_index(drop=True, inplace=True)

    # Compute moneyness
    clean_df["moneyness"] = clean_df["strike"] / spot_price

    return clean_df
