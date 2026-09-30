# Equity Volatility Analytics: SSVI Parameterisation & Microstructure Engine

Standard Black-Scholes pricing assumes constant volatility across all strike prices - a theoretical assumption consistently disproven by live options markets. 

This project goes beyond generic implied volatility plotting. It is a **Production-Grade Volatility Analytics Engine** designed to answer a rigorous quantitative question: *Given noisy market option quotes, how do we mathematically extract, clean, stitch, fit, and validate an arbitrage-free market-implied volatility surface?*

## Executive Summary

* **The Problem:** Raw options data is notoriously dirty (illiquid strikes, massive bid-ask spreads), and standard unweighted polynomial fits are highly vulnerable to both microstructure noise and calendar/butterfly arbitrage.
* **The Solution:** An end-to-end quantitative architecture that ingests intraday chain data, enforces fundamental no-arbitrage bounds, reverse-engineers Black-Scholes Implied Volatility (IV), and calibrates a **Surface Stochastic Volatility Inspired (SSVI)** model to the data.
* **Microstructure Immunity:** The SSVI calibration utilises an **Inverse-Variance (Spread-Weighted)** objective function, penalising illiquid OTM wings and ensuring the volatility curve tightly hugs the highly-liquid ATM quotes.

## Quantitative Architecture & Pipeline

The engine executes the following data flow sequentially:

1. **Intraday Data Ingestion:** Live yfinance API calls for option chains and underlying spot prices.
2. **Dynamic Risk-Free Rate Interpolation:** Queries live US Treasury tenors from the Federal Reserve (FRED) and fits a **Natural Cubic Spline** to extract the exact continuous risk-free rate r(T) for the option's maturity.
3. **Data Sanitisation:** Discards anomalous 0.00 bids and enforces fundamental no-arbitrage bounds.
4. **Implied Volatility Inversion:** Deploys a hybrid **Newton-Raphson / Brentq** root-finding algorithm to reverse-engineer IV from the observed mid-price.
5. **Forward Moneyness & OTM Stitching:** Calculates the Forward Price (F) to strip out risk-free drift and dividend decay, cleanly stitching OTM calls and OTM puts in Log-Forward Moneyness space.
6. **SSVI Calibration:** Fits the Gatheral & Jacquier (2014) SSVI formulation to the extracted smile.
7. **Arbitrage Diagnostics:** Analytically verifies Durrleman's Butterfly Arbitrage Condition on the calibrated slice.

## Tech Stack
* **Language:** Python
* **Quantitative Modelling:** scipy.optimize (Numerical root-finding, SLSQP constrained optimisation), scipy.interpolate (Cubic Splines)
* **Data Engineering:** pandas, 
umpy, yfinance, redapi
* **Frontend & Visualisation:** streamlit, plotly (Interactive microstructure and volatility charting)

## Access the Model

**1. Live Web Application (Recommended)**  
Access the interactive risk dashboard directly in your browser:  
[Launch Streamlit Dashboard](https://chileungmak-equity-vol-smile.streamlit.app/)

**2. Local Execution**  
To run the quantitative engine locally:
``bash
git clone https://github.com/chileungmak/equity-vol-smile.git
cd equity-vol-smile
pip install -r requirements.txt
python -m streamlit run app.py
``

## Author

**Chi Leung Mak (Ron), CFA, CAIA**  
*MSc Financial Engineering Candidate | Ex-Head of Business Analysis*  

Bridging alternative investments with quantitative financial modelling. Drawing on prior experience as a real estate research analyst and head of business analysis in tech consulting to build rigorous, data-driven analytical tools. 

[LinkedIn](https://linkedin.com/in/clmak) | [GitHub](https://github.com/chileungmak)
