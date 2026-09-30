import pytest
import numpy as np
import pandas as pd
from options_math import (
    black_scholes_price,
    black_scholes_vega,
    implied_volatility,
    calculate_chain_implied_volatility,
    fit_ssvi_slice
)
from data_pipeline import (
    clean_options_data
)

def test_black_scholes_price_atm():
    # ATM call and put should have predictable pricing
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    
    call_price = black_scholes_price(S, K, T, r, q, sigma, "call")
    put_price = black_scholes_price(S, K, T, r, q, sigma, "put")
    
    # Put-Call Parity: C - P = S*exp(-qT) - K*exp(-rT)
    expected_parity = S * np.exp(-q * T) - K * np.exp(-r * T)
    actual_parity = call_price - put_price
    
    assert np.isclose(actual_parity, expected_parity, atol=1e-5), "Put-Call parity violated"
    assert call_price > 0
    assert put_price > 0

def test_implied_volatility_roundtrip():
    S = 150.0
    K = 160.0
    T = 0.5
    r = 0.03
    q = 0.01
    true_sigma = 0.25
    
    # Generate price
    call_price = black_scholes_price(S, K, T, r, q, true_sigma, "call")
    
    # Recover sigma
    recovered_sigma = implied_volatility(
        price=call_price,
        S=S,
        K=K,
        T=T,
        r=r,
        q=q,
        option_type="call"
    )
    
    assert np.isclose(recovered_sigma, true_sigma, atol=1e-5), "IV recovery failed"

def test_vega_positivity():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20
    
    vega = black_scholes_vega(S, K, T, r, q, sigma)
    assert vega > 0, "Vega should be positive"

def test_clean_options_data():
    df = pd.DataFrame({
        "contractSymbol": ["TEST1", "TEST2"],
        "strike": [100.0, 110.0],
        "bid": [2.0, 0.0],
        "ask": [2.2, 0.5],
        "volume": [100, 50],
        "openInterest": [500, 200]
    })
    
    # S=100, K=100, T=1, r=0.05, q=0.0 -> Intrinsic for Call K=100 is S - K*exp(-rT) = 100 - 95.12 = 4.87
    # Wait, if S=100 and K=100, the mid price is 2.1. 
    # But intrinsic is 4.87, so the mid price (2.1) is BELOW intrinsic.
    # Therefore, it should be filtered out by the arbitrage filter!
    
    # Let's use a realistic OTM scenario so it survives
    # S=100, K=110, T=1, r=0.05, q=0.0. 
    # Call intrinsic = max(0, 100 - 110*exp(-rT)) = 0.
    # Mid price = (0.0 + 0.5) / 2 = 0.25 > 0.
    # BUT bid is 0.0, so if min_bid=0.01, it gets dropped.
    
    # Let's setup one that perfectly passes:
    df_pass = pd.DataFrame({
        "contractSymbol": ["PASS"],
        "strike": [110.0],
        "bid": [1.0],
        "ask": [1.2],
        "volume": [100],
        "openInterest": [100]
    })
    
    cleaned = clean_options_data(
        df_pass, 
        spot_price=100.0, 
        T=1.0, 
        r=0.05, 
        q=0.0, 
        option_type="call", 
        min_volume=10, 
        min_bid=0.5, 
        max_spread_pct=0.5
    )
    
    assert len(cleaned) == 1
    assert "mid_price" in cleaned.columns
    assert np.isclose(cleaned["mid_price"].iloc[0], 1.1)

def test_empty_chain_handling():
    # Simulate Yahoo Finance returning an empty chain or all options being filtered out
    empty_raw = pd.DataFrame(columns=["contractSymbol", "strike", "bid", "ask", "volume", "openInterest"])
    
    # 1. Cleaning should return empty without crashing
    cleaned = clean_options_data(
        empty_raw, spot_price=100.0, T=1.0, r=0.05, q=0.02, option_type="call"
    )
    assert cleaned.empty
    
    # 2. IV calculation should return empty without crashing
    iv_df = calculate_chain_implied_volatility(
        cleaned, S=100.0, T=1.0, r=0.05, q=0.02, option_type="call"
    )
    assert iv_df.empty
    
    # 3. Simulate SSVI fitting logic defensive checks
    dfs_to_concat = []
    if not iv_df.empty: dfs_to_concat.append(iv_df)
    fit_df = pd.concat(dfs_to_concat) if dfs_to_concat else pd.DataFrame()
    
    # Without the defensive check, fit_df.dropna(subset=["implied_volatility"]) would raise KeyError here.
    has_iv = "implied_volatility" in fit_df.columns
    assert not has_iv
    
    if not fit_df.empty and has_iv:
        fit_df = fit_df.dropna(subset=["implied_volatility"])
        
    assert fit_df.empty, "SSVI logic should gracefully bypass empty chains"



def test_ssvi_weighted_fit():
    k_array = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    w_array = np.array([0.06, 0.045, 0.04, 0.042, 0.05])
    
    res_unweighted = fit_ssvi_slice(k_array, w_array)
    assert res_unweighted is not None
    assert "theta" in res_unweighted
    assert res_unweighted["theta"] > 0
    
    weights = np.array([0.1, 0.1, 10.0, 0.1, 0.1])
    res_weighted = fit_ssvi_slice(k_array, w_array, weights=weights)
    
    assert res_weighted is not None
    assert np.isclose(res_weighted["theta"], 0.04, atol=1e-3)



