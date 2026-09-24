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
)
from options_math import (
    black_scholes_price,
    calculate_chain_implied_volatility,
)

# ---------------------------------------------------------
# Streamlit Page Configuration & Global Theming
# ---------------------------------------------------------
st.set_page_config(
    page_title="Options Volatility Smile Engine",
    page_icon="📈",
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
    overlay_yahoo_iv: bool = False,
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
) -> go.Figure:
    """Construct an interactive Plotly visualization of the Volatility Smile.

    Parameters
    ----------
    calls_df : pd.DataFrame
        DataFrame of processed Call contracts with solved IV.
    puts_df : pd.DataFrame
        DataFrame of processed Put contracts with solved IV.
    spot_price : float
        Current underlying asset spot price.
    x_axis_mode : str, default 'Strike Price ($)'
        'Strike Price ($)', 'Moneyness (K / S)', or 'Log-Moneyness ln(K / S)'.
    show_calls : bool, default True
        Whether to display Call IV trace.
    show_puts : bool, default True
        Whether to display Put IV trace.
    overlay_yahoo_iv : bool, default False
        Whether to overlay Yahoo Finance's uncleaned raw IV.
    strike_min : Optional[float]
        Lower strike cutoff for zoom.
    strike_max : Optional[float]
        Upper strike cutoff for zoom.

    Returns
    -------
    go.Figure
        Plotly Figure object displaying the volatility smile.
    """
    fig = go.Figure()

    def get_x_values(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
        if x_axis_mode == "Moneyness (K / S)":
            return df["strike"].values / spot_price, "Moneyness (K/S)"
        elif x_axis_mode == "Log-Moneyness ln(K / S)":
            return np.log(df["strike"].values / spot_price), "Log-Moneyness ln(K/S)"
        return df["strike"].values, "Strike Price ($)"

    # Filter by strike range if specified
    c_df = calls_df.copy()
    p_df = puts_df.copy()
    if strike_min is not None:
        c_df = c_df[c_df["strike"] >= strike_min]
        p_df = p_df[p_df["strike"] >= strike_min]
    if strike_max is not None:
        c_df = c_df[c_df["strike"] <= strike_max]
        p_df = p_df[p_df["strike"] <= strike_max]

    # Reference spot line coordinate on x-axis
    if x_axis_mode == "Moneyness (K / S)":
        spot_x = 1.0
        x_label = "Moneyness (K / S)"
    elif x_axis_mode == "Log-Moneyness ln(K / S)":
        spot_x = 0.0
        x_label = "Log-Moneyness ln(K / S)"
    else:
        spot_x = spot_price
        x_label = "Strike Price ($)"

    # Plot Calls
    if show_calls and not c_df.empty:
        x_c, _ = get_x_values(c_df)
        fig.add_trace(
            go.Scatter(
                x=x_c,
                y=c_df["iv_pct"],
                mode="lines+markers",
                name="Call IV (Reverse-Engineered)",
                line=dict(color="#00D4FF", width=2.5, shape="spline", smoothing=0.8),
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

        if overlay_yahoo_iv and "impliedVolatility" in c_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_c,
                    y=c_df["impliedVolatility"] * 100.0,
                    mode="markers",
                    name="Call Yahoo Raw IV",
                    marker=dict(size=4, color="#64B5F6", symbol="x", opacity=0.6),
                    hoverinfo="skip",
                )
            )

    # Plot Puts
    if show_puts and not p_df.empty:
        x_p, _ = get_x_values(p_df)
        fig.add_trace(
            go.Scatter(
                x=x_p,
                y=p_df["iv_pct"],
                mode="lines+markers",
                name="Put IV (Reverse-Engineered)",
                line=dict(color="#FF7043", width=2.5, shape="spline", smoothing=0.8),
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

    # Spot Price vertical reference marker
    fig.add_vline(
        x=spot_x,
        line_width=1.8,
        line_dash="dash",
        line_color="#E0E0E0",
        annotation_text=f"Spot: ${spot_price:.2f}" if x_axis_mode == "Strike Price ($)" else "ATM (Spot)",
        annotation_position="top right",
        annotation_font=dict(color="#E0E0E0", size=11),
    )

    # Polished layout
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


def main() -> None:
    """Main application loop and UI orchestration."""
    # ---------------------------------------------------------
    # Sidebar: Asset, Expiration & Quantitative Controls
    # ---------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Engine Controls")
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
        expirations: List[str] = meta["expirations"]

        st.success(f"**{ticker_input}** Spot: **${spot_price:,.2f} {meta['currency']}**")

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

        # 3. Macro Financial Parameters
        st.subheader("3. Macro Environment")
        risk_free_rate = st.number_input(
            "Risk-Free Rate (r)",
            min_value=0.00,
            max_value=0.20,
            value=0.045,
            step=0.0025,
            format="%.4f",
            help="Annualized continuously-compounded risk-free interest rate (e.g. 0.045 for 4.5%).",
        )

        # 4. Data Cleaning & Microstructure Filters
        with st.expander("🧹 Order Book Cleaning Filters", expanded=False):
            min_volume = st.number_input(
                "Min Volume",
                min_value=0,
                max_value=10000,
                value=1,
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
        with st.expander("🔬 Black-Scholes Solver Settings", expanded=False):
            solver_tol = st.select_slider(
                "Price Convergence Tolerance",
                options=[1e-4, 1e-5, 1e-6, 1e-7],
                value=1e-6,
                format_func=lambda x: f"{x:.1e}",
            )
            max_iterations = st.slider("Max Solver Iterations", 20, 200, 100, step=10)

        # 6. Cache Invalidation
        st.divider()
        if st.button("🔄 Refresh Data / Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ---------------------------------------------------------
    # Main Header & Dashboard Context
    # ---------------------------------------------------------
    st.title("📈 Options Volatility Smile Engine")
    st.markdown(
        "Production-grade Quantitative MVP: Ingests intraday options chains, sanitizes order book noise, "
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
        option_type="call",
        min_volume=int(min_volume),
        min_bid=float(min_bid),
        max_spread_pct=float(max_spread_pct),
        min_open_interest=int(min_open_interest),
    )

    puts_clean = clean_options_data(
        puts_raw,
        spot_price=spot_price,
        option_type="put",
        min_volume=int(min_volume),
        min_bid=float(min_bid),
        max_spread_pct=float(max_spread_pct),
        min_open_interest=int(min_open_interest),
    )

    # ---------------------------------------------------------
    # Numerical Root-Finding (Reverse-Engineer IV)
    # ---------------------------------------------------------
    with st.spinner("Executing Black-Scholes IV numerical root-finding (Newton-Raphson + Brentq)..."):
        calls_iv = calculate_chain_implied_volatility(
            calls_clean,
            S=spot_price,
            T=T,
            r=risk_free_rate,
            option_type="call",
            tol=solver_tol,
            max_iter=max_iterations,
        )
        puts_iv = calculate_chain_implied_volatility(
            puts_clean,
            S=spot_price,
            T=T,
            r=risk_free_rate,
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
                <div class="metric-sub">{ticker_input} • {meta['currency']}</div>
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
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">ATM Strike (K_ATM)</div>
                <div class="metric-value">{atm_val_str}</div>
                <div class="metric-sub">Distance: {abs(atm_strike - spot_price):.2f} pts</div>
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
    st.subheader("Volatility Smile Surface")

    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([1.5, 1.2, 1.2, 1.2])
    with c_ctrl1:
        x_mode = st.radio(
            "X-Axis Representation",
            ["Strike Price ($)", "Moneyness (K / S)", "Log-Moneyness ln(K / S)"],
            horizontal=True,
        )
    with c_ctrl2:
        show_calls = st.checkbox("Show Calls", value=True)
        show_puts = st.checkbox("Show Puts", value=True)
    with c_ctrl3:
        overlay_yahoo = st.checkbox(
            "Compare Yahoo Raw IV",
            value=False,
            help="Displays Yahoo's raw IV values as markers to contrast noise and artifacts against reverse-engineered mid-price IV.",
        )
    with c_ctrl4:
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

    # Render Plotly Smile Figure
    fig = build_volatility_smile_chart(
        calls_df=calls_iv,
        puts_df=puts_iv,
        spot_price=spot_price,
        x_axis_mode=x_mode,
        show_calls=show_calls,
        show_puts=show_puts,
        overlay_yahoo_iv=overlay_yahoo,
        strike_min=strike_range[0],
        strike_max=strike_range[1],
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Analytics & Diagnostics Tabs
    # ---------------------------------------------------------
    tab_data, tab_micro, tab_math = st.tabs(
        [
            "📊 Cleaned Options Chain Data",
            "🔍 Market Microstructure & Liquidity",
            "📐 Quantitative Math & Inversion Architecture",
        ]
    )

    with tab_data:
        st.caption("Cleaned, order-book sanitized options contracts with numerical IV solutions.")
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
                    label="📥 Download Calls Chain (CSV)",
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
                    label="📥 Download Puts Chain (CSV)",
                    data=csv_puts,
                    file_name=f"{ticker_input}_{selected_expiry}_puts_iv.csv",
                    mime="text/csv",
                )
            else:
                st.info("No Put options survived liquidity and no-arbitrage filtering.")

    with tab_micro:
        st.caption("Order book liquidity dynamics: Bid-Ask Spreads and Trading Volume by Strike.")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            # Spread % vs Strike
            fig_spread = go.Figure()
            if not calls_iv.empty:
                fig_spread.add_trace(
                    go.Scatter(
                        x=calls_iv["strike"],
                        y=calls_iv["spread_pct"] * 100.0,
                        mode="lines+markers",
                        name="Call Bid-Ask Spread %",
                        line=dict(color="#00D4FF", width=1.8),
                    )
                )
            if not puts_iv.empty:
                fig_spread.add_trace(
                    go.Scatter(
                        x=puts_iv["strike"],
                        y=puts_iv["spread_pct"] * 100.0,
                        mode="lines+markers",
                        name="Put Bid-Ask Spread %",
                        line=dict(color="#FF7043", width=1.8),
                    )
                )
            fig_spread.add_vline(x=spot_price, line_dash="dash", line_color="#8B949E")
            fig_spread.update_layout(
                title="Bid-Ask Spread (% of Mid-Price) vs Strike",
                xaxis_title="Strike ($)",
                yaxis_title="Spread %",
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#0D1117",
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
                        x=calls_iv["strike"],
                        y=calls_iv["volume"],
                        name="Call Volume",
                        marker_color="#00D4FF",
                        opacity=0.7,
                    )
                )
            if not puts_iv.empty:
                fig_vol.add_trace(
                    go.Bar(
                        x=puts_iv["strike"],
                        y=puts_iv["volume"],
                        name="Put Volume",
                        marker_color="#FF7043",
                        opacity=0.7,
                    )
                )
            fig_vol.add_vline(x=spot_price, line_dash="dash", line_color="#8B949E")
            fig_vol.update_layout(
                title="Trading Volume Distribution by Strike",
                xaxis_title="Strike ($)",
                yaxis_title="Contracts Traded",
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#0D1117",
                height=350,
                barmode="overlay",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    with tab_math:
        st.markdown(
            r"""
            ### Quantitative Mathematics: Black-Scholes Model Inversion

            #### 1. Theoretical Black-Scholes Pricing Framework
            For a non-dividend paying underlying asset with spot price $S$, strike $K$, annualized time to maturity $T$, risk-free rate $r$, and constant volatility $\sigma$:

            $$\begin{aligned}
            C(S, K, T, r, \sigma) &= S N(d_1) - K e^{-rT} N(d_2) \\
            P(S, K, T, r, \sigma) &= K e^{-rT} N(-d_2) - S N(-d_1)
            \end{aligned}$$

            where $N(x)$ denotes the standard normal cumulative distribution function (CDF), and:

            $$d_1 = \frac{\ln(S/K) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}$$

            #### 2. Reverse-Engineering Implied Volatility ($\sigma_{IV}$)
            Observed option market quotes trade at mid-price $P_{\text{market}} = \frac{\text{Bid} + \text{Ask}}{2}$. Implied Volatility is the unique parameter satisfying:

            $$f(\sigma) = \text{BS}(S, K, T, r, \sigma) - P_{\text{market}} = 0$$

            Because no closed-form analytical inverse exists for $N(d_1)$, we apply a two-tier numerical solver:

            1. **Primary Solver — Newton-Raphson Method**:
               Quadratic convergence via analytical Vega derivative $\nu = \frac{\partial \text{BS}}{\partial \sigma} = S \sqrt{T} \phi(d_1)$:
               $$\sigma_{n+1} = \sigma_n - \frac{\text{BS}(S, K, T, r, \sigma_n) - P_{\text{market}}}{\nu(S, K, T, r, \sigma_n)}$$

            2. **Fallback Solver — Brent's Root-Finding (`brentq`)**:
               If $\nu < 10^{-8}$ (vanishing Vega in deep OTM/ITM wings) or Newton steps outside $[\sigma_{\min}, \sigma_{\max}] = [0.01\%, 500\%]$, the solver falls back to Brent's algorithm, ensuring robust root bracketing without numerical instability.

            #### 3. No-Arbitrage Theoretical Boundaries
            In real-world markets, stale quotes and illiquidity can yield market prices violating lower or upper arbitrage boundaries:
            - **Call Boundary**: $\max(0, S - K e^{-rT}) < C_{\text{market}} < S$
            - **Put Boundary**: $\max(0, K e^{-rT} - S) < P_{\text{market}} < K e^{-rT}$

            Our pipeline sanitizes options chains before inversion, tagging non-convergent contracts and filtering out arbitrage violations.
            """
        )


if __name__ == "__main__":
    main()
