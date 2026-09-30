"""Options Volatility Smile Engine - Streamlit Application.

Interactive quantitative platform for ingesting intraday options chains,
sanitizing order book noise, reverse-engineering Black-Scholes implied volatilities
via hybrid numerical solvers, and visualizing the Volatility Smile & Surface.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import (
    calculate_time_to_expiration,
    clean_options_data,
    fetch_option_chain_raw,
    fetch_ticker_metadata,
    fetch_yield_curve,
)
from options_math import (
    black_scholes_price,
    calculate_chain_implied_volatility,
    build_yield_curve_spline,
    get_risk_free_rate,
    ssvi_total_variance,
    fit_ssvi_slice,
)

# ---------------------------------------------------------
# Streamlit Page Configuration & Global Theming
# ---------------------------------------------------------
st.set_page_config(
    page_title="Options Volatility Smile Engine",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-polish quant dashboard appearance
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F0F6FC;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #58A6FF;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_volatility_smile_chart(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    spot_price: float,
    x_axis_mode: str = "Strike Price ($)",
    show_calls: bool = True,
    show_puts: bool = True,
    stitch_otm: bool = True,
    overlay_yahoo_iv: bool = False,
    show_iv_spread: bool = True,
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
    ssvi_params: Optional[dict] = None,
    T: float = 1.0,
    r: float = 0.0,
    q: float = 0.0,
) -> go.Figure:
    """Construct an interactive Plotly visualization of the Volatility Smile."""
    fig = go.Figure()

    # Calculate Forward price for rigorous moneyness/stitching
    F = spot_price * np.exp((r - q) * T)

    def get_x_values(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
        if x_axis_mode == "Forward Moneyness (K / F)":
            return df["strike"].values / F, "Forward Moneyness (K / F)"
        elif x_axis_mode == "Log-Forward Moneyness ln(K / F)":
            return np.log(df["strike"].values / F), "Log-Forward Moneyness ln(K / F)"
        return df["strike"].values, "Strike Price ($)"

    # Copy and filter dataframes
    c_df = calls_df.copy()
    p_df = puts_df.copy()
    
    if stitch_otm:
        # Discard ITM options to avoid American early-exercise premium distortion
        # Stitch using forward price (F) rather than spot price (S)
        if not c_df.empty:
            c_df = c_df[c_df["strike"] >= F]
        if not p_df.empty:
            p_df = p_df[p_df["strike"] < F]

    if strike_min is not None:
        if not c_df.empty: c_df = c_df[c_df["strike"] >= strike_min]
        if not p_df.empty: p_df = p_df[p_df["strike"] >= strike_min]
    if strike_max is not None:
        if not c_df.empty: c_df = c_df[c_df["strike"] <= strike_max]
        if not p_df.empty: p_df = p_df[p_df["strike"] <= strike_max]

    # Reference Forward line coordinate on x-axis
    if x_axis_mode == "Forward Moneyness (K / F)":
        fwd_x = 1.0
        x_label = "Forward Moneyness (K / F)"
    elif x_axis_mode == "Log-Forward Moneyness ln(K / F)":
        fwd_x = 0.0
        x_label = "Log-Forward Moneyness ln(K / F)"
    else:
        fwd_x = F
        x_label = "Strike Price ($)"

    # Plot SSVI Arbitrage-Free Surface Curve
    if ssvi_params and ssvi_params.get("theta") is not None:
        F = spot_price * np.exp((r - q) * T)
        
        all_strikes = []
        if not c_df.empty: all_strikes.extend(c_df["strike"].tolist())
        if not p_df.empty: all_strikes.extend(p_df["strike"].tolist())
        
        if all_strikes:
            k_min = np.log(min(all_strikes) / F)
            k_max = np.log(max(all_strikes) / F)
            k_dense = np.linspace(k_min, k_max, 300)
            
            w_dense = ssvi_total_variance(k_dense, ssvi_params["theta"], ssvi_params["rho"], ssvi_params["phi"])
            iv_dense = np.sqrt(np.maximum(w_dense / T, 1e-8)) * 100.0
            
            strike_dense = F * np.exp(k_dense)
            
            if x_axis_mode == "Forward Moneyness (K / F)":
                x_dense = strike_dense / F
            elif x_axis_mode == "Log-Forward Moneyness ln(K / F)":
                x_dense = np.log(strike_dense / F)
            else:
                x_dense = strike_dense
                
            fig.add_trace(go.Scatter(
                x=x_dense, 
                y=iv_dense,
                mode="lines",
                name="SSVI Arbitrage-Free Fit",
                line=dict(color="#FF00FF", width=3, dash="solid"),
                hoverinfo="skip"
            ))

    # Plot Calls
    if show_calls and not c_df.empty:
        x_c, _ = get_x_values(c_df)
        fig.add_trace(
            go.Scatter(
                x=x_c,
                y=c_df["iv_pct"],
                mode="markers" if ssvi_params else "lines+markers",
                name="Call IV (Mid Price)",
                line=dict(color="#00D4FF", width=1.5, shape="spline", smoothing=0.8),
                marker=dict(size=6, color="#00D4FF", symbol="circle"),
                customdata=np.stack(
                    (
                        c_df["strike"],
                        c_df["mid_price"],
                        c_df["bid"],
                        c_df["ask"],
                        c_df["spread_pct"] * 100.0,
                        c_df["volume"],
                        c_df["openInterest"],
                        c_df.get("impliedVolatility", np.nan) * 100.0,
                    ),
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>Call Option</b><br>"
                    + "Strike: $%{customdata[0]:.2f}<br>"
                    + "Implied Volatility: <b>%{y:.2f}%</b><br>"
                    + "Mid Price: $%{customdata[1]:.2f} (Bid: $%{customdata[2]:.2f}, Ask: $%{customdata[3]:.2f})<br>"
                    + "Spread: %{customdata[4]:.1f}%<br>"
                    + "Volume: %{customdata[5]:,.0f} | OI: %{customdata[6]:,.0f}<br>"
                    + "Yahoo Raw IV: %{customdata[7]:.2f}%"
                    + "<extra></extra>"
                ),
            )
        )

        if show_iv_spread and "bid_iv_pct" in c_df.columns and "ask_iv_pct" in c_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_c, y=c_df["ask_iv_pct"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_c, y=c_df["bid_iv_pct"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(41, 182, 246, 0.15)",
                    line=dict(width=0),
                    name="Call IV Bid/Ask Spread",
                    hoverinfo="skip"
                )
            )

        if overlay_yahoo_iv and "impliedVolatility" in c_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_c,
                    y=c_df["impliedVolatility"] * 100.0,
                    mode="markers",
                    name="Call Yahoo Raw IV",
                    marker=dict(size=4, color="#81D4FA", symbol="x", opacity=0.6),
                    hoverinfo="skip",
                )
            )
    if show_puts and not p_df.empty:
        x_p, _ = get_x_values(p_df)
        fig.add_trace(
            go.Scatter(
                x=x_p,
                y=p_df["iv_pct"],
                mode="markers" if ssvi_params else "lines+markers",
                name="Put IV (Mid Price)",
                line=dict(color="#FF7043", width=1.5, shape="spline", smoothing=0.8),
                marker=dict(size=6, color="#FF7043", symbol="diamond"),
                customdata=np.stack(
                    (
                        p_df["strike"],
                        p_df["mid_price"],
                        p_df["bid"],
                        p_df["ask"],
                        p_df["spread_pct"] * 100.0,
                        p_df["volume"],
                        p_df["openInterest"],
                        p_df.get("impliedVolatility", np.nan) * 100.0,
                    ),
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>Put Option</b><br>"
                    + "Strike: $%{customdata[0]:.2f}<br>"
                    + "Implied Volatility: <b>%{y:.2f}%</b><br>"
                    + "Mid Price: $%{customdata[1]:.2f} (Bid: $%{customdata[2]:.2f}, Ask: $%{customdata[3]:.2f})<br>"
                    + "Spread: %{customdata[4]:.1f}%<br>"
                    + "Volume: %{customdata[5]:,.0f} | OI: %{customdata[6]:,.0f}<br>"
                    + "Yahoo Raw IV: %{customdata[7]:.2f}%"
                    + "<extra></extra>"
                ),
            )
        )

        if show_iv_spread and "bid_iv_pct" in p_df.columns and "ask_iv_pct" in p_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_p, y=p_df["ask_iv_pct"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_p, y=p_df["bid_iv_pct"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255, 112, 67, 0.15)",
                    line=dict(width=0),
                    name="Put IV Bid/Ask Spread",
                    hoverinfo="skip"
                )
            )

        if overlay_yahoo_iv and "impliedVolatility" in p_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_p,
                    y=p_df["impliedVolatility"] * 100.0,
                    mode="markers",
                    name="Put Yahoo Raw IV",
                    marker=dict(size=4, color="#FFAB91", symbol="x", opacity=0.6),
                    hoverinfo="skip",
                )
            )
    # Forward Price vertical reference marker
    fig.add_vline(
        x=fwd_x,
        line_width=1.8,
        line_dash="dash",
        line_color="#E0E0E0",
        annotation_text=f"Forward: " if x_axis_mode == "Strike Price ($)" else "ATM (Forward)",
        annotation_position="top right",
        annotation_font=dict(color="#E0E0E0", size=11),
    )

    # Polished layout
    x_range = None
    if strike_min is not None and strike_max is not None:
        if x_axis_mode == "Forward Moneyness (K / F)":
            x_range = [strike_min / F, strike_max / F]
        elif x_axis_mode == "Log-Forward Moneyness ln(K / F)":
            x_range = [float(np.log(strike_min / F)), float(np.log(strike_max / F))]
        else:
            x_range = [strike_min, strike_max]

    fig.update_layout(
        title=dict(
            text=f"Implied Volatility Smile ({x_label})",
            font=dict(size=18, color="#F0F6FC"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(color="#C9D1D9", size=13)),
            showgrid=True,
            gridcolor="#21262D",
            zeroline=False,
            color="#8B949E",
            range=x_range,
        ),
        yaxis=dict(
            title=dict(text="Implied Volatility (%)", font=dict(color="#C9D1D9", size=13)),
            showgrid=True,
            gridcolor="#21262D",
            zeroline=False,
            color="#8B949E",
            ticksuffix="%",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363D",
            borderwidth=1,
            font=dict(color="#C9D1D9"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        height=540,
        margin=dict(l=60, r=40, t=70, b=50),
        hovermode="closest",
    )

    return fig


def render_methodology_page():
    st.title("Quantitative Methodology & Architecture")
    st.markdown("This whitepaper details the mathematical and quantitative pipeline powering the options volatility engine.")
    
    st.header("1. Data Sanitisation & No-Arbitrage Bounds")
    st.markdown(r'''
Before root-finding, raw option quotes are sanitised to prevent solver divergence. We compute the mid-price as $P_{mid} = \frac{P_{bid} + P_{ask}}{2}$. 
Any quotes violating fundamental no-arbitrage bounds are discarded:
- **Calls:** $\max(0, S - K e^{-rT}) \leq C_{mkt} \leq S$
- **Puts:** $\max(0, K e^{-rT} - S) \leq P_{mkt} \leq K e^{-rT}$
    ''')
    
    st.header("2. Yield Curve Term Structure (Cubic Spline)")
    st.markdown(r'''
Standard option calculators use a static, hardcoded risk-free rate. This introduces maturity mismatch errors. 
Our engine dynamically interpolates the **US Treasury Yield Curve** using a **Natural Cubic Spline**:
1. We query live Constant Maturity Treasury rates (1M, 3M, 6M, 1Y, 2Y, etc.) from the Federal Reserve (FRED).
2. We fit a twice-differentiable piecewise polynomial $S(t)$ across the tenors.
3. For an option with Time to Expiration $T$, we extract the exact interpolated rate $r(T) = S(T)$ to compute the precise continuous discount factor $e^{-rT}$.
    ''')
    
    st.header("3. Forward Moneyness ($k$)")
    st.markdown(r'''
To rigorously model options and strip out the impact of risk-free drift and dividend decay, we convert the Spot Price ($S$) to the **Forward Price** ($F$):
$$ F = S e^{(r-q)T} $$
We stitch Out-of-the-Money (OTM) calls and puts precisely at $K = F$, and parameterise the volatility smile using log-forward moneyness ($k$):
$$ k = \ln\left(\frac{K}{F}\right) $$
    ''')

    st.header("4. Implied Volatility Inversion")
    st.markdown(r'''
We extract the Black-Scholes implied volatility $\sigma$ using a hybrid **Newton-Raphson / Brentq** root-finding algorithm.
The Newton step relies on the option's Vega ($\mathcal{V}$):
$$ \sigma_{n+1} = \sigma_n - \frac{C_{BS}(\sigma_n) - C_{mkt}}{\mathcal{V}(\sigma_n)} $$
If Vega vanishes (deep OTM options) or the Newton step escapes the search bounds, the algorithm automatically falls back to Brent's method for guaranteed convergence.
    ''')

    st.header("5. SSVI Parameterisation (Gatheral & Jacquier 2014)")
    st.markdown(r'''
We fit the **Surface Stochastic Volatility Inspired (SSVI)** model to the extracted volatility smile. The model parameterises the total implied variance $w(k) = \sigma^2(k) T$ as:
$$ w(k) = \frac{\theta}{2} \left[ 1 + \rho \phi k + \sqrt{(\phi k + \rho)^2 + (1 - \rho^2)} \right] $$
Where:
- $\theta$: ATM total variance ($w(0)$).
- $\rho$: Correlation parameter governing the **skew** ($-1 < \rho < 1$).
- $\phi$: Curvature parameter governing the **smile** ($\phi > 0$).
    ''')

    st.header("6. Arbitrage Diagnostics")
    st.markdown(r'''
Unlike generic polynomial splines, the SSVI formulation is analytically proven to prevent arbitrage. Our pipeline enforces Durrleman's **Butterfly Arbitrage Condition** on the fitted slice:
$$ \theta \phi (1 + |\rho|) \leq 4 $$
*(Note: Complete Calendar Arbitrage absence requires $\partial w / \partial T \geq 0$, which necessitates a multi-expiry surface calibration rather than a single slice).*
    ''')

    st.header("7. Weighted Calibration (Microstructure Noise)")
    st.markdown(r'''
Option markets contain illiquid quotes with massive Bid/Ask spreads deep in the wings. An unweighted Ordinary Least Squares (OLS) fit allows these noisy quotes to artificially distort the SSVI curvature ($\phi$). 
To immunize the calibration, we utilize a **Spread-Weighted Objective Function**, where the weight $W_i$ of each observation is inversely proportional to its spread variance:
$$ W_i \propto \frac{1}{(\text{Spread}_i)^2} $$
$$ \min_{\theta, \rho, \phi} \sum_{i} W_i \left(w_i - w_{SSVI}(k_i)\right)^2 $$
    ''')

def main() -> None:
    """Main application loop and UI orchestration."""
    # ---------------------------------------------------------
    # Sidebar: Asset, Expiration & Quantitative Controls
    # ---------------------------------------------------------
    with st.sidebar:
        app_mode = st.radio(
            "Navigation",
            ["Analytics Dashboard", "Quantitative Methodology"],
            label_visibility="collapsed"
        )
        st.divider()
        
        if app_mode == "Quantitative Methodology":
            st.info("View the full quantitative architecture and mathematical definitions used to power the Option Engine.")

    if app_mode == "Quantitative Methodology":
        render_methodology_page()
        return

    with st.sidebar:
        st.title("Engine Controls")
        st.caption("Quantitative Options Parameters")

        # 1. Ticker Selection
        st.subheader("1. Underlying Asset")
        popular_tickers = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA", "IWM"]
        selected_pill = st.pills("Quick Select", popular_tickers, default=None)

        ticker_input = st.text_input(
            "Ticker Symbol",
            value=selected_pill if selected_pill else "SPY",
            help="Enter a valid equity or ETF ticker (e.g., SPY, AAPL, NVDA).",
        ).strip().upper()

        # Fetch metadata with caching
        try:
            with st.spinner(f"Ingesting quotes for {ticker_input}..."):
                meta = fetch_ticker_metadata(ticker_input)
        except Exception as exc:
            st.error(f"Data Ingestion Error: {exc}")
            st.info("Please enter a valid ticker with active US options trading.")
            st.stop()

        spot_price = float(meta["spot_price"])
        dividend_yield = float(meta.get("dividend_yield", 0.0))
        expirations: List[str] = meta["expirations"]

        st.success(f"**{ticker_input}** Spot: **${spot_price:,.2f} {meta['currency']}** | Div Yield: **{dividend_yield * 100:.2f}%**")

        # 2. Expiration Selection
        st.subheader("2. Expiration Date")

        def format_expiry(d: str) -> str:
            try:
                days = (datetime.strptime(d, "%Y-%m-%d").date() - datetime.today().date()).days
                return f"{d} ({days} DTE)"
            except Exception:
                return d

        # Pick default: prefer ~15-45 DTE if available, otherwise first/second expiry
        default_index = 0
        if len(expirations) > 1:
            for idx, exp in enumerate(expirations):
                _, d = calculate_time_to_expiration(exp)
                if 10 <= d <= 45:
                    default_index = idx
                    break
            else:
                default_index = min(1, len(expirations) - 1)

        selected_expiry = st.selectbox(
            "Select Expiration",
            options=expirations,
            index=default_index,
            format_func=format_expiry,
        )

        T, dte = calculate_time_to_expiration(selected_expiry)

        # 3. Macro Financial Parameters (Dynamic Yield Curve)
        st.subheader("3. Macro Environment")
        
        try:
            with st.spinner("Fetching FRED Yield Curve..."):
                yc_tenors, yc_rates = fetch_yield_curve()
            yield_spline = build_yield_curve_spline(yc_tenors, yc_rates)
            risk_free_rate = get_risk_free_rate(yield_spline, T)
            
            mac1, mac2 = st.columns(2)
            mac1.metric(
                "Risk-Free Rate (r)",
                f"{risk_free_rate * 100:.3f}%",
                help=f"Dynamically interpolated from US Treasury spline for T={T:.3f} years."
            )
            mac2.metric(
                "Dividend Yield (q)",
                f"{dividend_yield * 100:.2f}%",
                help="Trailing annual continuous dividend yield."
            )
        except Exception as e:
            st.error("Yield Curve API error. Using fallback 4.5%.")
            risk_free_rate = 0.045
            st.metric("Fallback Risk-Free Rate", "4.500%")

        # 4. Data Cleaning & Microstructure Filters
        with st.expander("Order Book Cleaning Filters", expanded=False):
            min_volume = st.number_input(
                "Min Volume",
                min_value=0,
                max_value=10000,
                value=0,
                help="Eliminates zero-volume stale or ghost contracts.",
            )
            min_bid = st.number_input(
                "Min Bid Price ($)",
                min_value=0.0,
                max_value=50.0,
                value=0.05,
                step=0.01,
                help="Filters zero-bid contracts where market makers offer no bid.",
            )
            max_spread_pct = st.slider(
                "Max Bid-Ask Spread / Mid Price",
                min_value=0.05,
                max_value=1.50,
                value=0.60,
                step=0.05,
                help="Filters illiquid contracts with excessive bid-ask spreads relative to mid price.",
            )
            min_open_interest = st.number_input(
                "Min Open Interest",
                min_value=0,
                max_value=10000,
                value=0,
                help="Minimum open interest required.",
            )

        # 5. Solver Settings
        with st.expander("Black-Scholes Solver Settings", expanded=False):
            solver_tol = st.select_slider(
                "Price Convergence Tolerance",
                options=[1e-4, 1e-5, 1e-6, 1e-7],
                value=1e-6,
                format_func=lambda x: f"{x:.1e}",
            )
            max_iterations = st.slider("Max Solver Iterations", 20, 200, 100, step=10)

        # 6. Cache Invalidation
        st.divider()
        if st.button("Refresh Data / Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ---------------------------------------------------------
    # Main Header & Dashboard Context
    # ---------------------------------------------------------
    st.title("Options Volatility Smile Engine")
    st.markdown(
        "Production-grade Quantitative MVP: Ingests intraday options chains, sanitises order book noise, "
        "reverse-engineers the Black-Scholes pricing model to find Implied Volatility (IV), and maps the volatility smile."
    )

    # ---------------------------------------------------------
    # Ingestion & Cleaning Pipeline Execution
    # ---------------------------------------------------------
    try:
        with st.spinner(f"Ingesting raw option chain for {ticker_input} on {selected_expiry}..."):
            calls_raw, puts_raw = fetch_option_chain_raw(ticker_input, selected_expiry)
    except Exception as exc:
        st.error(f"Options Ingestion Failure: {exc}")
        st.stop()

    if calls_raw.empty and puts_raw.empty:
        st.warning(f"No option chain data available for {ticker_input} on {selected_expiry}.")
        st.stop()

    calls_clean = clean_options_data(
        calls_raw,
        spot_price=spot_price,
        T=T,
        r=risk_free_rate,
        q=dividend_yield,
        option_type="call",
        min_volume=int(min_volume),
        min_bid=float(min_bid),
        max_spread_pct=float(max_spread_pct),
        min_open_interest=int(min_open_interest),
    )

    puts_clean = clean_options_data(
        puts_raw,
        spot_price=spot_price,
        T=T,
        r=risk_free_rate,
        q=dividend_yield,
        option_type="put",
        min_volume=int(min_volume),
        min_bid=float(min_bid),
        max_spread_pct=float(max_spread_pct),
        min_open_interest=int(min_open_interest),
    )

    # ---------------------------------------------------------
    # Numerical Root-Finding (Reverse-Engineer IV)
    # ---------------------------------------------------------
    with st.spinner("Executing BSM IV numerical root-finding (Newton-Raphson + Brentq)..."):
        calls_iv = calculate_chain_implied_volatility(
            calls_clean,
            S=spot_price,
            T=T,
            r=risk_free_rate,
            q=dividend_yield,
            option_type="call",
            tol=solver_tol,
            max_iter=max_iterations,
        )
        puts_iv = calculate_chain_implied_volatility(
            puts_clean,
            S=spot_price,
            T=T,
            r=risk_free_rate,
            q=dividend_yield,
            option_type="put",
            tol=solver_tol,
            max_iter=max_iterations,
        )

    # ---------------------------------------------------------
    # Key Financial Metrics (KPI Cards)
    # ---------------------------------------------------------
    # Compute ATM strike and ATM IV
    all_strikes = sorted(
        list(set(calls_iv["strike"].tolist() + puts_iv["strike"].tolist()))
    )
    atm_strike: Optional[float] = None
    atm_call_iv: Optional[float] = None
    atm_put_iv: Optional[float] = None

    if all_strikes:
        atm_strike = min(all_strikes, key=lambda k: abs(k - spot_price))
        c_atm = calls_iv[calls_iv["strike"] == atm_strike]
        if not c_atm.empty:
            atm_call_iv = float(c_atm["iv_pct"].iloc[0])
        p_atm = puts_iv[puts_iv["strike"] == atm_strike]
        if not p_atm.empty:
            atm_put_iv = float(p_atm["iv_pct"].iloc[0])

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Underlying Spot</div>
                <div class="metric-value">${spot_price:,.2f}</div>
                <div class="metric-sub">{ticker_input} | {meta['currency']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Maturity / DTE</div>
                <div class="metric-value">{dte} Days</div>
                <div class="metric-sub">T = {T:.4f} yrs ({selected_expiry})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        atm_val_str = f"${atm_strike:.2f}" if atm_strike is not None else "N/A"
        dist_str = f"Distance: {abs(atm_strike - spot_price):.2f} pts" if atm_strike is not None else "Distance: N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">ATM Strike (K_ATM)</div>
                <div class="metric-value">{atm_val_str}</div>
                <div class="metric-sub">{dist_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        c_iv_str = f"{atm_call_iv:.2f}%" if atm_call_iv is not None else "N/A"
        p_iv_str = f"{atm_put_iv:.2f}%" if atm_put_iv is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">ATM Implied Vol</div>
                <div class="metric-value">{c_iv_str}</div>
                <div class="metric-sub">Call: {c_iv_str} | Put: {p_iv_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi5:
        total_raw = len(calls_raw) + len(puts_raw)
        total_solved = len(calls_iv) + len(puts_iv)
        conv_rate = (total_solved / max(1, len(calls_clean) + len(puts_clean))) * 100.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Order Book Filtered</div>
                <div class="metric-value">{total_solved} Strikes</div>
                <div class="metric-sub">Raw: {total_raw} | Conv: {conv_rate:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # Chart Control Bar & Interactive Plotly Volatility Smile
    # ---------------------------------------------------------
    st.subheader("Implied Volatility Smile (Slice)")

    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4, c_ctrl5 = st.columns([1.5, 1.2, 1.2, 1.4, 1.2])
    with c_ctrl1:
        x_mode = st.radio(
            "X-Axis",
            ["Strike Price ($)", "Forward Moneyness (K / F)", "Log-Forward Moneyness ln(K / F)"],
            horizontal=True,
        )
        calibration_weighting = st.selectbox(
            "SSVI Fit Weighting",
            options=["Unweighted (OLS)", "Spread-Weighted", "Vega-Weighted"],
            index=1,
            help="Weights for the SSVI calibration. Spread-Weighted reduces the influence of noisy, illiquid quotes."
        )
    with c_ctrl2:
        show_calls = st.checkbox("Show Calls", value=True)
        show_puts = st.checkbox("Show Puts", value=True)
    with c_ctrl3:
        overlay_yahoo = st.checkbox(
            "Compare Yahoo Raw IV",
            value=False,
            help="Displays Yahoo's raw IV values as markers to contrast noise against reverse-engineered mid-price IV.",
        )
        show_iv_spread = st.checkbox(
            "Show Bid/Ask IV Spread",
            value=True,
            help="Displays the IV bounds implied by the bid and ask quotes."
        )
    with c_ctrl4:
        stitch_otm = st.checkbox(
            "Stitch OTM Only",
            value=True,
            help="Eliminates 'American Premium' distortion by building the curve exclusively from OTM options."
        )
    with c_ctrl5:
        # Strike range filter slider
        if all_strikes:
            s_min_default = max(float(min(all_strikes)), spot_price * 0.75)
            s_max_default = min(float(max(all_strikes)), spot_price * 1.25)
            strike_range = st.slider(
                "Filter Strike Window",
                min_value=float(min(all_strikes)),
                max_value=float(max(all_strikes)),
                value=(s_min_default, s_max_default),
                step=1.0 if spot_price > 50 else 0.5,
            )
        else:
            strike_range = (None, None)

    # SSVI Fitting Logic (Gatheral & Jacquier 2014)
    ssvi_params = None
    with st.spinner("Fitting Surface SVI (SSVI) Arbitrage-Free Model..."):
        dfs_to_concat = []
        if not calls_iv.empty: dfs_to_concat.append(calls_iv)
        if not puts_iv.empty: dfs_to_concat.append(puts_iv)
        
        fit_df = pd.concat(dfs_to_concat) if dfs_to_concat else pd.DataFrame()
        
        F_val = spot_price * np.exp((risk_free_rate - dividend_yield) * T)

        if stitch_otm and not fit_df.empty:
            stitch_dfs = []
            if not calls_iv.empty:
                stitch_dfs.append(calls_iv[calls_iv["strike"] >= F_val])
            if not puts_iv.empty:
                stitch_dfs.append(puts_iv[puts_iv["strike"] < F_val])
            fit_df = pd.concat(stitch_dfs) if stitch_dfs else pd.DataFrame()
        
        if not fit_df.empty and "implied_volatility" in fit_df.columns:
            fit_df = fit_df.dropna(subset=["implied_volatility"])
            if len(fit_df) >= 4:
                k_arr = np.log(fit_df["strike"].values / F_val)
                w_arr = (fit_df["implied_volatility"].values ** 2) * T
                
                # Apply weighting scheme
                if calibration_weighting == "Spread-Weighted":
                    # Weight proportional to 1 / spread^2 (inverse variance). Adding 1e-4 to avoid div by zero.
                    weights = 1.0 / (np.maximum(fit_df["spread_pct"].values, 1e-4) ** 2)
                elif calibration_weighting == "Vega-Weighted":
                    weights = fit_df["vega"].values
                else:
                    weights = np.ones_like(w_arr)

                ssvi_params = fit_ssvi_slice(k_arr, w_arr, weights=weights)

    # Render Plotly Smile Figure
        fig = build_volatility_smile_chart(
        calls_df=calls_iv,
        puts_df=puts_iv,
        spot_price=spot_price,
        x_axis_mode=x_mode,
        show_calls=show_calls,
        show_puts=show_puts,
        stitch_otm=stitch_otm,
        overlay_yahoo_iv=overlay_yahoo,
        show_iv_spread=show_iv_spread,
        strike_min=strike_range[0],
        strike_max=strike_range[1],
        ssvi_params=ssvi_params,
        T=T,
        r=risk_free_rate,
        q=dividend_yield,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display SSVI Parameters
    if ssvi_params:
        with st.expander("SSVI Slice Fit Parameters & Diagnostics (Gatheral & Jacquier 2014)", expanded=True):
            st.markdown("Surface Stochastic Volatility Inspired (SSVI) fit parameterises the slice. Butterfly arbitrage condition is enforced. Calendar arbitrage requires multi-expiry calibration.")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Theta [ATM Variance]", f"{ssvi_params['theta']:.5f}")
            s2.metric("Rho [Correlation]", f"{ssvi_params['rho']:.5f}")
            s3.metric("Phi [Curvature]", f"{ssvi_params['phi']:.5f}")
            
            # Butterfly constraint: theta * phi * (1 + |rho|) <= 4
            bf_val = ssvi_params['theta'] * ssvi_params['phi'] * (1 + abs(ssvi_params['rho']))
            bf_pass = bf_val <= 4.00001
            s4.metric(
                "Butterfly Condition",
                "PASS" if bf_pass else "FAIL",
                help=f"theta * phi * (1 + |rho|) = {bf_val:.4f} (Must be <= 4)"
            )

    # ---------------------------------------------------------
    # Analytics & Diagnostics Tabs
    # ---------------------------------------------------------
    tab_data, tab_micro, tab_yield = st.tabs([
        "Cleaned Options Chain Data",
        "Market Microstructure & Liquidity",
        "Yield Curve Term Structure"
    ])

    with tab_data:
        st.caption("Cleaned, order-book sanitised options contracts with numerical IV solutions.")
        sub_tab_calls, sub_tab_puts = st.tabs(["Calls Chain", "Puts Chain"])

        display_cols = [
            "strike",
            "bid",
            "ask",
            "mid_price",
            "spread_pct",
            "volume",
            "openInterest",
            "iv_pct",
            "impliedVolatility",
            "solver_error",
        ]

        with sub_tab_calls:
            if not calls_iv.empty:
                cols_present = [c for c in display_cols if c in calls_iv.columns]
                c_display = calls_iv[cols_present].copy()
                c_display["spread_pct"] = (c_display["spread_pct"] * 100.0).round(2)
                c_display["iv_pct"] = c_display["iv_pct"].round(2)
                if "impliedVolatility" in c_display.columns:
                    c_display["yahoo_iv_pct"] = (c_display["impliedVolatility"] * 100.0).round(2)
                    c_display.drop(columns=["impliedVolatility"], inplace=True)
                if "solver_error" in c_display.columns:
                    c_display["solver_error"] = c_display["solver_error"].apply(lambda e: f"{e:.2e}")

                st.dataframe(
                    c_display.style.format(
                        {
                            "strike": "${:.2f}",
                            "bid": "${:.2f}",
                            "ask": "${:.2f}",
                            "mid_price": "${:.2f}",
                            "spread_pct": "{:.1f}%",
                            "iv_pct": "{:.2f}%",
                            "yahoo_iv_pct": "{:.2f}%",
                            "volume": "{:,.0f}",
                            "openInterest": "{:,.0f}",
                        },
                        na_rep="-",
                    ),
                    use_container_width=True,
                    height=350,
                )
                csv_calls = calls_iv.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Calls Chain (CSV)",
                    data=csv_calls,
                    file_name=f"{ticker_input}_{selected_expiry}_calls_iv.csv",
                    mime="text/csv",
                )
            else:
                st.info("No Call options survived liquidity and no-arbitrage filtering.")

        with sub_tab_puts:
            if not puts_iv.empty:
                cols_present = [c for c in display_cols if c in puts_iv.columns]
                p_display = puts_iv[cols_present].copy()
                p_display["spread_pct"] = (p_display["spread_pct"] * 100.0).round(2)
                p_display["iv_pct"] = p_display["iv_pct"].round(2)
                if "impliedVolatility" in p_display.columns:
                    p_display["yahoo_iv_pct"] = (p_display["impliedVolatility"] * 100.0).round(2)
                    p_display.drop(columns=["impliedVolatility"], inplace=True)
                if "solver_error" in p_display.columns:
                    p_display["solver_error"] = p_display["solver_error"].apply(lambda e: f"{e:.2e}")

                st.dataframe(
                    p_display.style.format(
                        {
                            "strike": "${:.2f}",
                            "bid": "${:.2f}",
                            "ask": "${:.2f}",
                            "mid_price": "${:.2f}",
                            "spread_pct": "{:.1f}%",
                            "iv_pct": "{:.2f}%",
                            "yahoo_iv_pct": "{:.2f}%",
                            "volume": "{:,.0f}",
                            "openInterest": "{:,.0f}",
                        },
                        na_rep="-",
                    ),
                    use_container_width=True,
                    height=350,
                )
                csv_puts = puts_iv.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Puts Chain (CSV)",
                    data=csv_puts,
                    file_name=f"{ticker_input}_{selected_expiry}_puts_iv.csv",
                    mime="text/csv",
                )
            else:
                st.info("No Put options survived liquidity and no-arbitrage filtering.")

    with tab_micro:
        st.caption("Order book liquidity dynamics: Bid-Ask Spreads and Trading Volume by Strike.")
        
        # Determine X-Axis metrics for microstructure charts
        F_val = spot_price * np.exp((risk_free_rate - dividend_yield) * T)
        if x_mode == "Forward Moneyness (K / F)":
            fwd_x_micro = 1.0
            micro_x_label = "Forward Moneyness (K / F)"
            c_x = calls_iv["strike"] / F_val if not calls_iv.empty else []
            p_x = puts_iv["strike"] / F_val if not puts_iv.empty else []
        elif x_mode == "Log-Forward Moneyness ln(K / F)":
            fwd_x_micro = 0.0
            micro_x_label = "Log-Forward Moneyness ln(K / F)"
            c_x = np.log(calls_iv["strike"] / F_val) if not calls_iv.empty else []
            p_x = np.log(puts_iv["strike"] / F_val) if not puts_iv.empty else []
        else:
            fwd_x_micro = F_val
            micro_x_label = "Strike Price ($)"
            c_x = calls_iv["strike"] if not calls_iv.empty else []
            p_x = puts_iv["strike"] if not puts_iv.empty else []

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            # Spread % vs Strike
            fig_spread = go.Figure()
            if not calls_iv.empty:
                fig_spread.add_trace(
                    go.Scatter(
                        x=c_x,
                        y=calls_iv["spread_pct"] * 100.0,
                        mode="lines+markers",
                        name="Call Bid-Ask Spread %",
                        line=dict(color="#00D4FF", width=1.8),
                    )
                )
            if not puts_iv.empty:
                fig_spread.add_trace(
                    go.Scatter(
                        x=p_x,
                        y=puts_iv["spread_pct"] * 100.0,
                        mode="lines+markers",
                        name="Put Bid-Ask Spread %",
                        line=dict(color="#FF7043", width=1.8),
                    )
                )
            fig_spread.add_vline(x=fwd_x_micro, line_dash="dash", line_color="#8B949E")
            fig_spread.update_layout(
                title=dict(text=f"Bid-Ask Spread (% of Mid-Price) vs {micro_x_label.split()[0]}", font=dict(size=16, color="#F0F6FC"), x=0.01, y=0.96),
                xaxis=dict(title=dict(text=micro_x_label, font=dict(color="#C9D1D9", size=13)), showgrid=True, gridcolor="#21262D", zeroline=False, color="#8B949E"),
                yaxis=dict(title=dict(text="Spread %", font=dict(color="#C9D1D9", size=13)), showgrid=True, gridcolor="#21262D", zeroline=False, color="#8B949E"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(22, 27, 34, 0.8)", bordercolor="#30363D", borderwidth=1, font=dict(color="#C9D1D9")),
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#0D1117",
                font=dict(color="#E6EDF3"),
                height=350,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_spread, use_container_width=True)

        with col_m2:
            # Volume profile
            fig_vol = go.Figure()
            if not calls_iv.empty:
                fig_vol.add_trace(
                    go.Bar(
                        x=c_x,
                        y=calls_iv["volume"],
                        name="Call Volume",
                        marker_color="#00D4FF",
                        opacity=0.7,
                    )
                )
            if not puts_iv.empty:
                fig_vol.add_trace(
                    go.Bar(
                        x=p_x,
                        y=puts_iv["volume"],
                        name="Put Volume",
                        marker_color="#FF7043",
                        opacity=0.7,
                    )
                )
            fig_vol.add_vline(x=fwd_x_micro, line_dash="dash", line_color="#8B949E")
            fig_vol.update_layout(
                title=dict(text=f"Trading Volume Distribution by {micro_x_label.split()[0]}", font=dict(size=16, color="#F0F6FC"), x=0.01, y=0.96),
                xaxis=dict(title=dict(text=micro_x_label, font=dict(color="#C9D1D9", size=13)), showgrid=True, gridcolor="#21262D", zeroline=False, color="#8B949E"),
                yaxis=dict(title=dict(text="Contracts Traded", font=dict(color="#C9D1D9", size=13)), showgrid=True, gridcolor="#21262D", zeroline=False, color="#8B949E"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(22, 27, 34, 0.8)", bordercolor="#30363D", borderwidth=1, font=dict(color="#C9D1D9")),
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#0D1117",
                font=dict(color="#E6EDF3"),
                height=350,
                barmode="overlay",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    with tab_yield:
        st.caption("Live US Treasury yield curve fetched from FRED, interpolated via natural Cubic Spline.")
        
        try:
            fig_yc = go.Figure()
            
            # Generate smooth points for spline curve
            t_smooth = np.linspace(yc_tenors[0], yc_tenors[-1], 200)
            r_smooth = yield_spline(t_smooth)
            
            # Plot continuous spline
            fig_yc.add_trace(
                go.Scatter(
                    x=t_smooth,
                    y=r_smooth * 100.0,
                    mode="lines",
                    name="Cubic Spline",
                    line=dict(color="#FF7043", width=2, shape="spline"),
                    hoverinfo="skip"
                )
            )
            
            # Plot FRED data points
            fig_yc.add_trace(
                go.Scatter(
                    x=yc_tenors,
                    y=yc_rates * 100.0,
                    mode="markers",
                    name="FRED Data (Constant Maturity)",
                    marker=dict(color="#00D4FF", size=8),
                    hovertemplate="Tenor: %{x:.2f} yrs<br>Yield: %{y:.3f}%<extra></extra>"
                )
            )
            
            # Plot exact interpolated rate for selected option
            fig_yc.add_trace(
                go.Scatter(
                    x=[T],
                    y=[risk_free_rate * 100.0],
                    mode="markers",
                    name=f"Selected Expiry (T={T:.3f})",
                    marker=dict(color="#00FF00", size=12, symbol="star"),
                    hovertemplate="<b>Selected Option Maturity</b><br>Tenor: %{x:.3f} yrs<br>Interpolated Rate: %{y:.3f}%<extra></extra>"
                )
            )
            
            fig_yc.update_layout(
                title=dict(text="Dynamic Risk-Free Rate Interpolation", font=dict(size=16, color="#F0F6FC"), x=0.01, y=0.96),
                xaxis=dict(title=dict(text="Tenor (Years)", font=dict(color="#C9D1D9", size=13)), showgrid=True, gridcolor="#21262D", zeroline=False, color="#8B949E"),
                yaxis=dict(title=dict(text="Yield (%)", font=dict(color="#C9D1D9", size=13)), showgrid=True, gridcolor="#21262D", zeroline=False, color="#8B949E"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(22, 27, 34, 0.8)", bordercolor="#30363D", borderwidth=1, font=dict(color="#C9D1D9")),
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#0D1117",
                font=dict(color="#E6EDF3"),
                height=450,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_yc, use_container_width=True)
            
            st.info(f"**Selected Rate**: The numerical solver is currently using **{risk_free_rate * 100:.3f}%** to evaluate the {selected_expiry} option chain.")
        except Exception as e:
            st.warning("Yield curve plot is currently unavailable.")


if __name__ == "__main__":
    main()




