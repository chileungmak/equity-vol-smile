"""Options Financial Mathematics Engine.

Provides analytical Black-Scholes pricing, Greeks (Vega), and a robust
hybrid numerical root-finder (Newton-Raphson with Brentq fallback) to reverse-engineer
Implied Volatility (IV) from market mid-prices.
"""

from typing import Literal, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize
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
    q: float,
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
    q : float
        Annualized continuously-compounded dividend yield.
    sigma : float
        Annualized volatility of the underlying asset (sigma > 0).
    option_type : Literal['call', 'put'], default 'call'
        Type of option contract.

    Returns
    -------
    float
        Theoretical Black-Scholes-Merton option price.
    """
    if S <= 0.0 or K <= 0.0:
        return np.nan

    S_adj = S * np.exp(-q * T)
    K_adj = K * np.exp(-r * T)

    # Zero or negative expiration boundary: return immediate payoff
    if T <= 0.0:
        if option_type == "call":
            return max(0.0, S - K)
        return max(0.0, K - S)

    # Zero or negative volatility boundary: return discounted intrinsic value
    if sigma <= 0.0:
        if option_type == "call":
            return max(0.0, S_adj - K_adj)
        return max(0.0, K_adj - S_adj)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        price = S_adj * norm.cdf(d1) - K_adj * norm.cdf(d2)
    elif option_type == "put":
        price = K_adj * norm.cdf(-d2) - S_adj * norm.cdf(-d1)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'.")

    return float(max(0.0, price))


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Compute the Black-Scholes-Merton Vega (dPrice / dSigma).

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
    q : float
        Annualized dividend yield.
    sigma : float
        Annualized volatility.

    Returns
    -------
    float
        Option Vega (dollar change in option value per 100% change in volatility).
    """
    if S <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return 0.0

    S_adj = S * np.exp(-q * T)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(S_adj * sqrt_T * norm.pdf(d1))


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
    sigma_min: float = 1e-4,
    sigma_max: float = 5.0,
) -> float:
    """Reverse-engineer Black-Scholes-Merton Implied Volatility using Newton-Raphson with Brentq fallback.

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
    q : float
        Annualized dividend yield.
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

    S_adj = S * np.exp(-q * T)
    K_adj = K * np.exp(-r * T)

    # Arbitrage bounds check:
    if option_type == "call":
        intrinsic = max(0.0, S_adj - K_adj)
        upper_bound = S_adj
    elif option_type == "put":
        intrinsic = max(0.0, K_adj - S_adj)
        upper_bound = K_adj
    else:
        return np.nan

    # If market price is at or below intrinsic, or above upper theoretical bound,
    # no valid positive finite implied volatility exists.
    if price <= intrinsic or price >= upper_bound:
        return np.nan

    # Initial volatility estimate (Brenner-Subrahmanyam approximation anchored near ATM)
    sigma_guess = np.sqrt(2.0 * np.pi / T) * (price / S_adj)
    sigma = float(np.clip(sigma_guess, 0.10, 1.50))

    # Phase 1: Newton-Raphson iteration
    for _ in range(max_iter):
        bs_p = black_scholes_price(S, K, T, r, q, sigma, option_type)
        diff = bs_p - price

        if abs(diff) < tol:
            return float(sigma)

        vega = black_scholes_vega(S, K, T, r, q, sigma)
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
            return black_scholes_price(S, K, T, r, q, sig, option_type) - price

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
    q: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> pd.DataFrame:
    if df.empty or "strike" not in df.columns or "mid_price" not in df.columns:
        return df.copy()

    res = df.copy()
    ivs, bid_ivs, ask_ivs, pricing_errors, vegas = [], [], [], [], []

    for _, row in res.iterrows():
        k = float(row["strike"])
        p = float(row["mid_price"])
        bid = float(row.get("bid", 0.0))
        ask = float(row.get("ask", 0.0))

        iv = implied_volatility(p, S, k, T, r, q, option_type, tol, max_iter)
        ivs.append(iv)
        
        b_iv = implied_volatility(bid, S, k, T, r, q, option_type, tol, max_iter) if bid > 0 else np.nan
        a_iv = implied_volatility(ask, S, k, T, r, q, option_type, tol, max_iter) if ask > 0 else np.nan
        bid_ivs.append(b_iv)
        ask_ivs.append(a_iv)

        if not np.isnan(iv):
            recomputed = black_scholes_price(S, k, T, r, q, iv, option_type)
            pricing_errors.append(abs(recomputed - p))
            vegas.append(black_scholes_vega(S, k, T, r, q, iv))
        else:
            pricing_errors.append(np.nan)
            vegas.append(np.nan)

    res["implied_volatility"] = ivs
    res["iv_pct"] = res["implied_volatility"] * 100.0
    res["bid_iv"] = bid_ivs
    res["ask_iv"] = ask_ivs
    res["bid_iv_pct"] = res["bid_iv"] * 100.0
    res["ask_iv_pct"] = res["ask_iv"] * 100.0
    res["vega"] = vegas
    
    F = S * np.exp((r - q) * T)
    res["moneyness"] = res["strike"] / S
    res["forward_moneyness"] = res["strike"] / F
    res["log_moneyness"] = np.log(res["forward_moneyness"])
    res["solver_error"] = pricing_errors
    res["option_type"] = option_type

    res_clean = res.dropna(subset=["implied_volatility"]).copy()
    res_clean.sort_values(by="strike", inplace=True)
    res_clean.reset_index(drop=True, inplace=True)

    return res_clean

def ssvi_total_variance(k: np.ndarray, theta: float, rho: float, phi: float) -> np.ndarray:
    """Calculate SSVI total implied variance w(k) based on Gatheral & Jacquier (2014).
    
    Parameters
    ----------
    k : np.ndarray
        Log-moneyness k = ln(K / F).
    theta : float
        ATM total variance.
    rho : float
        Correlation parameter (-1 < rho < 1).
    phi : float
        Smile curvature parameter (phi > 0).
        
    Returns
    -------
    np.ndarray
        Total variance w(k).
    """
    inner = (phi * k + rho)**2 + (1.0 - rho**2)
    inner = np.maximum(inner, 0.0) # Safety clip for floating point precision
    return (theta / 2.0) * (1.0 + rho * phi * k + np.sqrt(inner))


def fit_ssvi_slice(k_array: np.ndarray, w_array: np.ndarray, weights: np.ndarray = None) -> dict:
    """Fit SSVI parameters (theta, rho, phi) to a single volatility slice using constrained optimisation.
    
    Enforces the SSVI butterfly arbitrage-free condition: theta * phi * (1 + |rho|) <= 4
    
    Parameters
    ----------
    k_array : np.ndarray
        Array of log-moneyness.
    w_array : np.ndarray
        Array of total implied variance (sigma^2 * T).
    weights : np.ndarray, optional
        Weights for each observation. If None, equal weighting is used.
        
    Returns
    -------
    dict
        Dictionary containing 'theta', 'rho', 'phi', and 'mse' if successful, else None.
    """
    if len(k_array) < 4:
        return None
        
    # Remove NaNs
    valid = ~np.isnan(k_array) & ~np.isnan(w_array)
    k_array = k_array[valid]
    w_array = w_array[valid]
    if weights is not None:
        weights = weights[valid]
    else:
        weights = np.ones_like(w_array)
        
    if len(k_array) < 4:
        return None

    # Normalize weights to sum to N so the MSE scaling remains comparable
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights) * len(weights)
    else:
        weights = np.ones_like(w_array)

    # Initial guesses
    idx_atm = np.argmin(np.abs(k_array))
    theta_guess = max(float(w_array[idx_atm]), 1e-4)
    rho_guess = 0.0
    phi_guess = 1.0
    
    def objective(params):
        theta, rho, phi = params
        w_fit = ssvi_total_variance(k_array, theta, rho, phi)
        # Scale MSE heavily to prevent premature SLSQP convergence on tiny variance gradients
        return np.average((w_array - w_fit)**2, weights=weights) * 1e6
        
    # Bounds: theta > 0, -1 < rho < 1, phi > 0
    bounds = ((1e-5, 2.0), (-0.999, 0.999), (1e-5, 100.0))
    
    # SSVI Butterfly Arbitrage-Free Condition: 4 - theta * phi * (1 + |rho|) >= 0
    def constraint(params):
        theta, rho, phi = params
        return 4.0 - theta * phi * (1.0 + abs(rho))
        
    cons = {'type': 'ineq', 'fun': constraint}
    
    try:
        res = minimize(
            objective, 
            x0=[theta_guess, rho_guess, phi_guess],
            bounds=bounds,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if res.success:
            return {
                'theta': float(res.x[0]),
                'rho': float(res.x[1]),
                'phi': float(res.x[2]),
                'mse': float(res.fun)
            }
    except Exception:
        pass
        
    return None



