"""Options Financial Mathematics Engine.

Provides analytical Black-Scholes pricing, Greeks (Vega), and a robust
hybrid numerical root-finder (Newton-Raphson with Brentq fallback) to reverse-engineer
Implied Volatility (IV) from market mid-prices.
"""

from typing import Literal, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm
import streamlit as st

OptionType = Literal["call", "put"]


def build_yield_curve_spline(tenors: np.ndarray, rates: np.ndarray) -> CubicSpline:
    """Build a cubic spline interpolation for the yield curve.

    Parameters
    ----------
    tenors : np.ndarray
        Array of term-structure tenors in years.
    rates : np.ndarray
        Array of corresponding risk-free rates (decimals).

    Returns
    -------
    CubicSpline
        A callable cubic spline object.
    """
    return CubicSpline(tenors, rates, bc_type='natural')


def get_risk_free_rate(spline: CubicSpline, T: float) -> float:
    """Extract the interpolated risk-free rate for a specific time to maturity.
    
    Clamps the time to maturity to the min/max known tenors to avoid wild
    polynomial extrapolation artifacts at the boundaries.

    Parameters
    ----------
    spline : CubicSpline
        The fitted yield curve spline.
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        Interpolated risk-free rate.
    """
    T_clamped = np.clip(T, spline.x[0], spline.x[-1])
    return float(spline(T_clamped))


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Calculate the theoretical Black-Scholes-Merton European option price.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset (S > 0).
    K : float
        Strike price of the option (K > 0).
    T : float
        Time to expiration in years (T >= 0).
    r : float
        Annualized continuously-compounded risk-free interest rate (e.g., 0.045 for 4.5%).
    sigma : float
        Annualized volatility of the underlying asset (sigma > 0).
    option_type : Literal['call', 'put'], default 'call'
        Type of option contract.

    Returns
    -------
    float
        Theoretical Black-Scholes option price.
    """
    if S <= 0.0 or K <= 0.0:
        return np.nan

    # Zero or negative expiration boundary: return immediate payoff
    if T <= 0.0:
        if option_type == "call":
            return max(0.0, S - K)
        return max(0.0, K - S)

    # Zero or negative volatility boundary: return discounted intrinsic value
    if sigma <= 0.0:
        discounted_strike = K * np.exp(-r * T)
        if option_type == "call":
            return max(0.0, S - discounted_strike)
        return max(0.0, discounted_strike - S)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'.")

    return float(max(0.0, price))


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Compute the Black-Scholes Vega (dPrice / dSigma).

    Vega is identical for European Call and Put options under standard assumptions.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to expiration in years.
    r : float
        Annualized risk-free interest rate.
    sigma : float
        Annualized volatility.

    Returns
    -------
    float
        Option Vega (dollar change in option value per 100% change in volatility).
    """
    if S <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(S * sqrt_T * norm.pdf(d1))


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
    sigma_min: float = 1e-4,
    sigma_max: float = 5.0,
) -> float:
    """Reverse-engineer Black-Scholes Implied Volatility using Newton-Raphson with Brentq fallback.

    Parameters
    ----------
    price : float
        Observed market price (typically mid_price = (bid + ask) / 2).
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Annualized risk-free interest rate.
    option_type : Literal['call', 'put'], default 'call'
        Option type.
    tol : float, default 1e-6
        Convergence tolerance on option price difference.
    max_iter : int, default 100
        Maximum iterations for Newton-Raphson.
    sigma_min : float, default 1e-4 (0.01%)
        Minimum allowable volatility search bound.
    sigma_max : float, default 5.0 (500%)
        Maximum allowable volatility search bound.

    Returns
    -------
    float
        Implied volatility as an annualized standard deviation (e.g., 0.22 for 22%),
        or np.nan if inversion fails or violates arbitrage bounds.
    """
    if price <= 0.0 or S <= 0.0 or K <= 0.0 or T <= 0.0:
        return np.nan

    # Arbitrage bounds check:
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(0.0, S - discounted_strike)
        upper_bound = S
    elif option_type == "put":
        intrinsic = max(0.0, discounted_strike - S)
        upper_bound = discounted_strike
    else:
        return np.nan

    # If market price is at or below intrinsic, or above upper theoretical bound,
    # no valid positive finite implied volatility exists.
    if price <= intrinsic or price >= upper_bound:
        return np.nan

    # Initial volatility estimate (Brenner-Subrahmanyam approximation anchored near ATM)
    sigma_guess = np.sqrt(2.0 * np.pi / T) * (price / S)
    sigma = float(np.clip(sigma_guess, 0.10, 1.50))

    # Phase 1: Newton-Raphson iteration
    for _ in range(max_iter):
        bs_p = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = bs_p - price

        if abs(diff) < tol:
            return float(sigma)

        vega = black_scholes_vega(S, K, T, r, sigma)
        # Avoid division by zero or extremely flat gradient in deep wings
        if vega < 1e-8:
            break

        sigma_new = sigma - diff / vega

        # If Newton-Raphson steps out of search domain or oscillates wildly, switch to Brent's method
        if sigma_new <= sigma_min or sigma_new >= sigma_max:
            break

        sigma = sigma_new

    # Phase 2: Robust fallback via Brent's method (scipy.optimize.brentq)
    try:
        def objective(sig: float) -> float:
            return black_scholes_price(S, K, T, r, sig, option_type) - price

        f_min = objective(sigma_min)
        f_max = objective(sigma_max)

        # Roots must bracket zero for Brent's method
        if f_min * f_max <= 0.0:
            sol = brentq(objective, sigma_min, sigma_max, xtol=tol, maxiter=max_iter)
            return float(sol)
    except Exception:
        pass

    return np.nan


@st.cache_data(show_spinner=False)
def calculate_chain_implied_volatility(
    df: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> pd.DataFrame:
    """Calculate implied volatility across an entire options chain DataFrame.

    Cached via Streamlit `@st.cache_data` to ensure instantaneous UI interactions.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned options DataFrame containing at least 'strike' and 'mid_price' columns.
    S : float
        Current spot price of the underlying asset.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate.
    option_type : Literal['call', 'put']
        Option type.
    tol : float, default 1e-6
        Convergence tolerance for solver.
    max_iter : int, default 100
        Maximum iterations for solver.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with 'implied_volatility', 'moneyness',
        'log_moneyness', and solver convergence metadata.
    """
    if df.empty or "strike" not in df.columns or "mid_price" not in df.columns:
        return df.copy()

    res = df.copy()
    ivs = []
    pricing_errors = []

    for _, row in res.iterrows():
        k = float(row["strike"])
        p = float(row["mid_price"])
        iv = implied_volatility(
            price=p,
            S=S,
            K=k,
            T=T,
            r=r,
            option_type=option_type,
            tol=tol,
            max_iter=max_iter,
        )
        ivs.append(iv)

        if not np.isnan(iv):
            recomputed = black_scholes_price(S, k, T, r, iv, option_type)
            pricing_errors.append(abs(recomputed - p))
        else:
            pricing_errors.append(np.nan)

    res["implied_volatility"] = ivs
    res["iv_pct"] = res["implied_volatility"] * 100.0
    res["moneyness"] = res["strike"] / S
    res["log_moneyness"] = np.log(res["moneyness"])
    res["solver_error"] = pricing_errors
    res["option_type"] = option_type

    # Filter out rows where the solver could not converge or arbitrage boundaries were hit
    res_clean = res.dropna(subset=["implied_volatility"]).copy()
    res_clean.sort_values(by="strike", inplace=True)
    res_clean.reset_index(drop=True, inplace=True)

    return res_clean
