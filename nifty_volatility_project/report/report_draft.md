# Project 1 — NIFTY Volatility & Return Forecasting Dashboard
### Internship Project Report — NSE

*This draft was auto-generated from results_log.json. Expand each section with your own analysis and conclusions before submitting.*

## 01 Fetch
**NIFTY50** — Raw daily OHLCV, 1234 rows, 2021-07-12 to 2026-07-10
- rows: 1234

**NIFTYBANK** — Raw daily OHLCV, 1233 rows, 2021-07-12 to 2026-07-10
- rows: 1233

**NIFTYIT** — Raw daily OHLCV, 1233 rows, 2021-07-12 to 2026-07-10
- rows: 1233

**NIFTYAUTO** — Raw daily OHLCV, 1223 rows, 2021-07-12 to 2026-07-03
- rows: 1223

## 02 Clean
**NIFTY50** — Cleaned series with log returns, 1233 rows, 0 missing values remaining
- rows: 1233
- missing_values: 0

**NIFTYBANK** — Cleaned series with log returns, 1232 rows, 0 missing values remaining
- rows: 1232
- missing_values: 0

**NIFTYIT** — Cleaned series with log returns, 1232 rows, 0 missing values remaining
- rows: 1232
- missing_values: 0

**NIFTYAUTO** — Cleaned series with log returns, 1222 rows, 0 missing values remaining
- rows: 1222
- missing_values: 0

## 03 Eda
**ALL** — Rolling 21-day annualized volatility across all 4 indices - look for clustering (calm vs turbulent periods), not a flat line
![Rolling 21-day annualized volatility across all 4 indices - look for clustering (calm vs turbulent periods), not a flat line](..\figures\01_rolling_volatility.png)

**ALL** — Return distributions - fat tails (kurtosis > 0) are the justification for GARCH
![Return distributions - fat tails (kurtosis > 0) are the justification for GARCH](..\figures\02_return_distributions.png)

**NIFTY50** — Skew and excess kurtosis of daily log returns
- skew: -0.471
- excess_kurtosis: 3.975

**NIFTYBANK** — Skew and excess kurtosis of daily log returns
- skew: -0.558
- excess_kurtosis: 5.521

**NIFTYIT** — Skew and excess kurtosis of daily log returns
- skew: -0.24
- excess_kurtosis: 2.202

**NIFTYAUTO** — Skew and excess kurtosis of daily log returns
- skew: -0.181
- excess_kurtosis: 2.42

**ALL** — Cross-index return correlation - confirms the 4 indices add genuine variety, not near-duplicate series
![Cross-index return correlation - confirms the 4 indices add genuine variety, not near-duplicate series](..\figures\03_correlation_matrix.png)

## 04 Arima
**NIFTY50** — ARIMA(1,0,1) walk-forward forecast, RMSE=0.00832, MAE=0.00615
![ARIMA(1,0,1) walk-forward forecast, RMSE=0.00832, MAE=0.00615](..\figures\04_arima_NIFTY50.png)

**NIFTY50** — ARIMA(1,0,1) walk-forward return forecast
- rmse: 0.00832
- mae: 0.006151

**NIFTYBANK** — ARIMA(1,0,1) walk-forward forecast, RMSE=0.01041, MAE=0.00705
![ARIMA(1,0,1) walk-forward forecast, RMSE=0.01041, MAE=0.00705](..\figures\04_arima_NIFTYBANK.png)

**NIFTYBANK** — ARIMA(1,0,1) walk-forward return forecast
- rmse: 0.010411
- mae: 0.007052

**NIFTYIT** — ARIMA(1,0,1) walk-forward forecast, RMSE=0.01558, MAE=0.01118
![ARIMA(1,0,1) walk-forward forecast, RMSE=0.01558, MAE=0.01118](..\figures\04_arima_NIFTYIT.png)

**NIFTYIT** — ARIMA(1,0,1) walk-forward return forecast
- rmse: 0.015576
- mae: 0.011183

**NIFTYAUTO** — ARIMA(1,0,1) walk-forward forecast, RMSE=0.01328, MAE=0.00958
![ARIMA(1,0,1) walk-forward forecast, RMSE=0.01328, MAE=0.00958](..\figures\04_arima_NIFTYAUTO.png)

**NIFTYAUTO** — ARIMA(1,0,1) walk-forward return forecast
- rmse: 0.013279
- mae: 0.00958

## 05 Garch
**NIFTY50** — GARCH(1,1) volatility forecast vs realized, RMSE=0.45636
![GARCH(1,1) volatility forecast vs realized, RMSE=0.45636](..\figures\05_garch_NIFTY50.png)

**NIFTY50** — GARCH(1,1) volatility forecast vs 5-day realized volatility
- rmse_vs_realized_vol: 0.45636

**NIFTYBANK** — GARCH(1,1) volatility forecast vs realized, RMSE=0.65159
![GARCH(1,1) volatility forecast vs realized, RMSE=0.65159](..\figures\05_garch_NIFTYBANK.png)

**NIFTYBANK** — GARCH(1,1) volatility forecast vs 5-day realized volatility
- rmse_vs_realized_vol: 0.651588

**NIFTYIT** — GARCH(1,1) volatility forecast vs realized, RMSE=0.72365
![GARCH(1,1) volatility forecast vs realized, RMSE=0.72365](..\figures\05_garch_NIFTYIT.png)

**NIFTYIT** — GARCH(1,1) volatility forecast vs 5-day realized volatility
- rmse_vs_realized_vol: 0.723648

**NIFTYAUTO** — GARCH(1,1) volatility forecast vs realized, RMSE=0.66739
![GARCH(1,1) volatility forecast vs realized, RMSE=0.66739](..\figures\05_garch_NIFTYAUTO.png)

**NIFTYAUTO** — GARCH(1,1) volatility forecast vs 5-day realized volatility
- rmse_vs_realized_vol: 0.667392

## 06 Ml Benchmark
**NIFTY50** — Random Forest (5 lags) forecast, RMSE=0.00844, MAE=0.00627
![Random Forest (5 lags) forecast, RMSE=0.00844, MAE=0.00627](..\figures\06_rf_NIFTY50.png)

**NIFTY50** — Random Forest (5 lagged returns) forecast
- rmse: 0.008436
- mae: 0.006268

**NIFTYBANK** — Random Forest (5 lags) forecast, RMSE=0.01039, MAE=0.00712
![Random Forest (5 lags) forecast, RMSE=0.01039, MAE=0.00712](..\figures\06_rf_NIFTYBANK.png)

**NIFTYBANK** — Random Forest (5 lagged returns) forecast
- rmse: 0.01039
- mae: 0.007119

**NIFTYIT** — Random Forest (5 lags) forecast, RMSE=0.01562, MAE=0.01136
![Random Forest (5 lags) forecast, RMSE=0.01562, MAE=0.01136](..\figures\06_rf_NIFTYIT.png)

**NIFTYIT** — Random Forest (5 lagged returns) forecast
- rmse: 0.015623
- mae: 0.011359

**NIFTYAUTO** — Random Forest (5 lags) forecast, RMSE=0.01328, MAE=0.00953
![Random Forest (5 lags) forecast, RMSE=0.01328, MAE=0.00953](..\figures\06_rf_NIFTYAUTO.png)

**NIFTYAUTO** — Random Forest (5 lagged returns) forecast
- rmse: 0.013277
- mae: 0.009527

## 07 Comparison
**ALL** — Head-to-head RMSE comparison, ARIMA vs Random Forest, across all 4 indices
![Head-to-head RMSE comparison, ARIMA vs Random Forest, across all 4 indices](..\figures\07_model_comparison.png)
