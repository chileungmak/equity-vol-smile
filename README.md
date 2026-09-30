# Equity Derivatives Risk: Implied Volatility Smile & Market Microstructure Engine

Standard Black-Scholes pricing assumes constant volatility across all strike prices—a theoretical assumption consistently disproven by live options markets. This project constructs an automated pipeline to ingest intraday options chains, sanitise order book noise, and mathematically invert the Black-Scholes formula to visualize the real-world Volatility Smile (skew and kurtosis) across liquid equities.

## 📊 Executive Summary
* **The Problem:** Raw options data is notoriously dirty (illiquid strikes, zero bids, massive bid-ask spreads), and standard theoretical models fail to account for how the market actually prices tail risk.
* **The Solution:** An end-to-end quantitative architecture that dynamically ingests live chain data, filters market microstructure anomalies, and establishes clean mid-price equivalents for institutional-grade analysis.
* **Quantitative Inversion:** A custom numerical root-finding algorithm reverse-engineers the Black-Scholes pricing model to calculate precise Implied Volatility (IV) for every valid strike, mapping both Call and Put skews simultaneously.

## ⚙️ Methodology & Financial Engineering

### 1. Data Architecture & Microstructure Cleaning
* **Live Data Feed:** Intraday options chains and underlying spot prices are ingested dynamically via the `yfinance` API.
* **Sanitisation Pipeline:** The engine automatically strips zero-volume contracts, drops anomalous $0.00 bids, and calculates a true `mid_price` `((bid + ask) / 2)`. This prevents the root-finding algorithm from crashing or returning `NaN` on highly illiquid deep Out-of-the-Money (OTM) strikes.
* **State Management:** Implements `@st.cache_data` to cache expensive API payloads and numerical calculations, ensuring a zero-latency, highly responsive user interface without triggering API rate limits.

### 2. Quantitative Math (Black-Scholes Inversion)
* **Model Formulation:** Implements the core Black-Scholes-Merton equations for European calls and puts.
* **Iterative Solver:** Because Implied Volatility cannot be isolated algebraically from the Black-Scholes equation, the engine deploys a computationally optimised iterative root-finding solver (via `scipy.optimize`) to calculate the exact volatility that equates the theoretical price to the observed market mid-price.
* **Moneyness Normalisation:** Beyond standard Strike Price ($), the engine dynamically maps the X-axis into Moneyness ($K/S$) and Log-Moneyness ($\ln(K/S)$) spaces. This allows for standardized volatility skew analysis across different asset classes and expiration horizons.

## 💻 Tech Stack
* **Language:** Python
* **Quantitative Modeling:** `scipy` (numerical root-finding), custom Black-Scholes mathematics
* **Data Manipulation:** `pandas`, `numpy`, `yfinance`
* **Frontend & Deployment:** `streamlit`, `plotly` (interactive surface rendering), Streamlit Community Cloud

## 🚀 Access the Model

**1. Live Web Application (Recommended)**  
Access the interactive risk dashboard directly in your browser:  
👉 [Launch Streamlit Dashboard](https://chileungmak-equity-vol-smile.streamlit.app/)

**2. Local Execution**  
To run the model locally, clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```


## 👤 Author

**Chi Leung Mak (Ron), CFA, CAIA**  
*MSc Financial Engineering Candidate | Ex-Head of Business Analysis*  

Bridging alternative investment with quantitative financial modelling. Drawing on prior experience as a real estate research analyst and head of business analysis in tech consulting to build rigorous, data-driven analytical tools. 

[LinkedIn](https://linkedin.com/in/clmak) • [GitHub](https://github.com/chileungmak)


