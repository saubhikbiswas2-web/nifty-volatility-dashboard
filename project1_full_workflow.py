import matplotlib
matplotlib.use('Agg')
# %% [markdown]
# # Project 1 — NIFTY Volatility & Return Forecasting Dashboard
# ### Full workflow, one file, run cell-by-cell in VS Code (Shift+Enter on each `# %%` block)
#
# **How this file works:** every `# %%` starts a new cell. VS Code's Jupyter extension runs
# each cell in its own interactive window, so you get notebook-style execution — see a chart,
# inspect a dataframe, keep going — without Colab. You can also just run `python
# project1_full_workflow.py` in a terminal and it executes top to bottom as a normal script.
#
# **The output-logging system (read this before you run anything):** every meaningful result —
# a chart, a metric, a forecast — gets saved to disk AND appended to `results/results_log.json`
# through two helper functions, `save_fig()` and `log_result()`. At the end, `generate_report_draft()`
# reads that log and writes a draft report in Markdown with every chart and number already placed
# in order. You never have to remember "which RMSE was that again" — it's all logged the moment
# it's computed. This is what makes the final report foolproof: nothing depends on your memory.
#
# **Folder structure this creates, right next to this script:**
# ```
# nifty_volatility_project/
#   data/raw/        <- untouched data straight from the source
#   data/clean/       <- cleaned series + returns, ready for modeling
#   figures/          <- every chart, PNG, consistently named
#   results/          <- every metrics table (CSV) + the running results_log.json
#   models/           <- (optional) saved model objects
#   report/           <- report_draft.md, auto-generated at the end
# ```

# %%
import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
sns.set_style('darkgrid')

ROOT = Path("nifty_volatility_project")
DIRS = {
    "raw": ROOT / "data" / "raw",
    "clean": ROOT / "data" / "clean",
    "figures": ROOT / "figures",
    "results": ROOT / "results",
    "models": ROOT / "models",
    "report": ROOT / "report",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

LOG_PATH = DIRS["results"] / "results_log.json"


def log_result(stage, index_name, description, metrics=None, artifact_path=None):
    """Appends one entry to results_log.json. Call this every time you compute
    something worth remembering - a metric, a saved chart, a finding."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "index": index_name,
        "description": description,
        "metrics": metrics or {},
        "artifact": str(artifact_path) if artifact_path else None,
    }
    log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"[LOGGED] {stage} | {index_name} | {description}")


def save_fig(fig, filename, stage, index_name, description):
    """Saves a matplotlib figure to /figures with a consistent name AND logs it."""
    path = DIRS["figures"] / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log_result(stage, index_name, description, artifact_path=path)
    return path


print("Project folders ready at:", ROOT.resolve())

# %% [markdown]
# ## Stage 1 — Fetch the data
#
# **Why yfinance as the primary source:** NSE's own historical-data page blocks scripted
# requests without careful session-header handling. yfinance mirrors the same index values
# and is fully scriptable. Best practice: also manually export the same date range from
# NSE's site this week (nseindia.com/reports-indices-historical-index-data) and eyeball
# that the two agree — then your report can honestly say "cross-checked against NSE."
#
# **Why these 4 indices:** NIFTY 50 (the benchmark) plus 3 sectoral indices (Bank, IT, Auto)
# gives genuine variety — one index alone risks your model comparison being a lucky fit
# rather than a real finding.

# %%
tickers = {
    "NIFTY50":   "^NSEI",
    "NIFTYBANK": "^NSEBANK",
    "NIFTYIT":   "^CNXIT",
    "NIFTYAUTO": "^CNXAUTO",
}

raw_data = {}
for name, ticker in tickers.items():
    print(f"Fetching {name} ({ticker}) ...")
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        print(f"  WARNING: no data for {ticker} - verify the symbol on finance.yahoo.com")
        continue
    raw_data[name] = df
    df.to_csv(DIRS["raw"] / f"{name}_raw.csv")
    log_result("01_fetch", name, f"Raw daily OHLCV, {len(df)} rows, "
               f"{df.index.min().date()} to {df.index.max().date()}",
               metrics={"rows": len(df)},
               artifact_path=DIRS["raw"] / f"{name}_raw.csv")

print("\nFetched and saved:", list(raw_data.keys()))

# %% [markdown]
# ## Stage 2 — Clean & compute log returns
#
# **Why log returns:** they're additive across time periods, which is exactly what the
# ARIMA and GARCH models below assume. This is a standard, defensible modeling choice -
# be ready to explain it if your mentor asks.
#
# **Why we don't fill calendar gaps:** those are market holidays, not missing data.
# We only check for genuinely missing values inside what yfinance actually returned.

# %%
clean = {}
for name, df in raw_data.items():
    d = df[['Close']].rename(columns={'Close': 'close'}).dropna()
    d['log_ret'] = np.log(d['close'] / d['close'].shift(1))
    d = d.dropna(subset=['log_ret'])
    clean[name] = d
    d.to_csv(DIRS["clean"] / f"{name}_clean.csv")
    log_result("02_clean", name, f"Cleaned series with log returns, {len(d)} rows, "
               f"{d.isna().sum().sum()} missing values remaining",
               metrics={"rows": len(d), "missing_values": int(d.isna().sum().sum())},
               artifact_path=DIRS["clean"] / f"{name}_clean.csv")

returns_df = pd.DataFrame({name: d['log_ret'] for name, d in clean.items()}).dropna()
print(returns_df.describe())

# %% [markdown]
# ## Stage 3 — EDA
#
# This is what your later modeling choices rest on - you're checking, visually, whether
# the assumptions those models make (volatility clustering, fat tails) actually hold,
# before you fit anything.

# %%
# 3a - Rolling 21-day annualized volatility (21 trading days ~ 1 calendar month, standard convention)
fig, ax = plt.subplots(figsize=(11, 5))
for name, d in clean.items():
    rolling_vol = d['log_ret'].rolling(21).std() * np.sqrt(252) * 100
    ax.plot(rolling_vol.index, rolling_vol, label=name)
ax.set_title("21-Day Rolling Annualized Volatility (%)")
ax.set_ylabel("Annualized Volatility (%)")
ax.legend()
plt.tight_layout()
save_fig(fig, "01_rolling_volatility.png", "03_eda", "ALL",
         "Rolling 21-day annualized volatility across all 4 indices - "
         "look for clustering (calm vs turbulent periods), not a flat line")
plt.show()

# %%
# 3b - Return distributions + skew/kurtosis (the empirical case for GARCH over simpler models)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
kurt_summary = {}
for ax, (name, d) in zip(axes.flatten(), clean.items()):
    sns.histplot(d['log_ret'], bins=80, kde=True, ax=ax, color='seagreen')
    ax.set_title(f"{name} — Daily Log Return Distribution")
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    kurt_summary[name] = {"skew": round(float(d['log_ret'].skew()), 3),
                            "excess_kurtosis": round(float(d['log_ret'].kurtosis()), 3)}
plt.tight_layout()
save_fig(fig, "02_return_distributions.png", "03_eda", "ALL",
         "Return distributions - fat tails (kurtosis > 0) are the justification for GARCH")
plt.show()

for name, stats in kurt_summary.items():
    log_result("03_eda", name, "Skew and excess kurtosis of daily log returns", metrics=stats)

# %%
# 3c - Correlation across indices
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(returns_df.corr(), annot=True, cmap='YlGnBu', vmin=0, vmax=1, ax=ax)
ax.set_title("Correlation of Daily Returns Across Indices")
plt.tight_layout()
save_fig(fig, "03_correlation_matrix.png", "03_eda", "ALL",
         "Cross-index return correlation - confirms the 4 indices add genuine variety, "
         "not near-duplicate series")
plt.show()

# %% [markdown]
# ## Stage 4 — ARIMA (models the return series itself)
#
# **Why ARIMA(1,0,1):** a simple, standard starting specification for daily equity returns -
# one autoregressive term, one moving-average term, no differencing needed since log returns
# are already stationary. You can experiment with other orders later; log every attempt.
#
# **Why walk-forward, not a single train/test split:** predicting one step at a time and
# rolling forward mimics how you'd actually use this model in production - forecast tomorrow,
# see the actual outcome, refit, forecast the next day. A single static split overstates
# how good the model looks.

# %%
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

arima_results = {}
for name, d in clean.items():
    series = d['log_ret']
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]

    history = list(train)
    preds = []
    for actual in test:
        model = ARIMA(history, order=(1, 0, 1)).fit()
        preds.append(model.forecast(1)[0])
        history.append(actual)  # walk forward: reveal the true value, refit next step

    rmse = mean_squared_error(test, preds) ** 0.5
    mae = mean_absolute_error(test, preds)
    arima_results[name] = {"rmse": rmse, "mae": mae}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test.index, test.values, label="Actual", alpha=0.7)
    ax.plot(test.index, preds, label="ARIMA(1,0,1) Forecast", alpha=0.8)
    ax.set_title(f"{name} — ARIMA Walk-Forward Forecast vs Actual")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, f"04_arima_{name}.png", "04_arima", name,
             f"ARIMA(1,0,1) walk-forward forecast, RMSE={rmse:.5f}, MAE={mae:.5f}",
             )
    plt.show()

    log_result("04_arima", name, "ARIMA(1,0,1) walk-forward return forecast",
               metrics={"rmse": round(rmse, 6), "mae": round(mae, 6)})

pd.DataFrame(arima_results).T.to_csv(DIRS["results"] / "arima_metrics.csv")
print(pd.DataFrame(arima_results).T)

# %% [markdown]
# ## Stage 5 — GARCH (models volatility, not the returns themselves)
#
# **Why GARCH(1,1):** the standard baseline volatility model - today's variance depends on
# yesterday's variance and yesterday's squared shock. This is precisely testing what Stage 3's
# fat-tail finding predicted you'd need.

# %%
from arch import arch_model

garch_results = {}
for name, d in clean.items():
    series = d['log_ret'] * 100  # arch expects returns scaled to roughly %, for numerical stability
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]

    am = arch_model(train, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')

    forecast = res.forecast(horizon=len(test), reindex=False)
    predicted_vol = np.sqrt(forecast.variance.values[-1, :])
    realized_vol = test.rolling(5).std().dropna()  # 5-day realized vol as the comparison benchmark

    n = min(len(predicted_vol), len(realized_vol))
    rmse = mean_squared_error(realized_vol.values[:n], predicted_vol[:n]) ** 0.5
    garch_results[name] = {"rmse_vs_realized_vol": rmse}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(realized_vol.index[:n], realized_vol.values[:n], label="Realized Volatility (5d)", alpha=0.7)
    ax.plot(realized_vol.index[:n], predicted_vol[:n], label="GARCH(1,1) Forecast", alpha=0.8)
    ax.set_title(f"{name} — GARCH(1,1) Forecast vs Realized Volatility")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, f"05_garch_{name}.png", "05_garch", name,
             f"GARCH(1,1) volatility forecast vs realized, RMSE={rmse:.5f}")
    plt.show()

    log_result("05_garch", name, "GARCH(1,1) volatility forecast vs 5-day realized volatility",
               metrics={"rmse_vs_realized_vol": round(rmse, 6)})

pd.DataFrame(garch_results).T.to_csv(DIRS["results"] / "garch_metrics.csv")
print(pd.DataFrame(garch_results).T)

# %% [markdown]
# ## Stage 6 — ML benchmark (Random Forest on lagged returns)
#
# **Why include this at all:** your project's actual analytical point is comparing an
# econometric approach (ARIMA/GARCH) against a machine-learning approach on the same
# problem - that comparison IS the "AI + econometrics" story for your CV, not just an
# extra model for its own sake.

# %%
from sklearn.ensemble import RandomForestRegressor

N_LAGS = 5
rf_results = {}
for name, d in clean.items():
    series = d['log_ret']
    feat_df = pd.DataFrame({f"lag_{i}": series.shift(i) for i in range(1, N_LAGS + 1)})
    feat_df['target'] = series
    feat_df = feat_df.dropna()

    split = int(len(feat_df) * 0.8)
    X_train, y_train = feat_df.iloc[:split, :-1], feat_df.iloc[:split, -1]
    X_test, y_test = feat_df.iloc[split:, :-1], feat_df.iloc[split:, -1]

    rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    rf_results[name] = {"rmse": rmse, "mae": mae}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.index, y_test.values, label="Actual", alpha=0.7)
    ax.plot(y_test.index, preds, label="Random Forest Forecast", alpha=0.8)
    ax.set_title(f"{name} — Random Forest Forecast vs Actual")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, f"06_rf_{name}.png", "06_ml_benchmark", name,
             f"Random Forest (5 lags) forecast, RMSE={rmse:.5f}, MAE={mae:.5f}")
    plt.show()

    log_result("06_ml_benchmark", name, "Random Forest (5 lagged returns) forecast",
               metrics={"rmse": round(rmse, 6), "mae": round(mae, 6)})

pd.DataFrame(rf_results).T.to_csv(DIRS["results"] / "rf_metrics.csv")
print(pd.DataFrame(rf_results).T)

# %% [markdown]
# ## Stage 7 — Model comparison (the analytical centerpiece of the whole project)

# %%
comparison = pd.DataFrame({
    "ARIMA_RMSE": {k: v["rmse"] for k, v in arima_results.items()},
    "RF_RMSE": {k: v["rmse"] for k, v in rf_results.items()},
})
comparison.to_csv(DIRS["results"] / "model_comparison.csv")

fig, ax = plt.subplots(figsize=(9, 5))
comparison.plot(kind="bar", ax=ax)
ax.set_title("Forecast RMSE by Model and Index (lower is better)")
ax.set_ylabel("RMSE")
plt.tight_layout()
save_fig(fig, "07_model_comparison.png", "07_comparison", "ALL",
         "Head-to-head RMSE comparison, ARIMA vs Random Forest, across all 4 indices")
plt.show()

print(comparison)

# %% [markdown]
# ## Stage 8 — Auto-generate the report draft
#
# This reads everything logged above and writes a Markdown draft with every chart and
# metric already placed in the order you produced them. Open `report/report_draft.md`,
# expand each section with your own interpretation, and you have a complete, sourced
# internship report - nothing recalled from memory, everything traceable to a saved file.

# %%
def generate_report_draft():
    log = json.loads(LOG_PATH.read_text())
    lines = [
        "# Project 1 — NIFTY Volatility & Return Forecasting Dashboard",
        "### Internship Project Report — NSE",
        "",
        "*This draft was auto-generated from results_log.json. Expand each section with your own analysis and conclusions before submitting.*",
        "",
    ]
    seen_stages = []
    for e in log:
        if e["stage"] not in seen_stages:
            seen_stages.append(e["stage"])

    for stage in seen_stages:
        lines.append(f"## {stage.replace('_', ' ').title()}")
        for e in [x for x in log if x["stage"] == stage]:
            lines.append(f"**{e['index']}** — {e['description']}")
            for k, v in e["metrics"].items():
                lines.append(f"- {k}: {v}")
            if e["artifact"] and str(e["artifact"]).endswith(".png"):
                rel = os.path.relpath(e["artifact"], DIRS["report"])
                lines.append(f"![{e['description']}]({rel})")
            lines.append("")

    report_path = DIRS["report"] / "report_draft.md"
    report_path.write_text("\n".join(lines))
    print(f"Report draft written to: {report_path.resolve()}")
    print("Open it in VS Code (it will render the images) and start writing your commentary around it.")

generate_report_draft()

# %% [markdown]
# ## What you have at this point
# - `data/raw` and `data/clean` — every stage of the data, never overwritten
# - `figures/` — 9+ charts, consistently named, presentation-ready
# - `results/` — every metrics table as CSV, plus the master `results_log.json`
# - `report/report_draft.md` — a structured draft with every chart and number already placed
#
# **Next (Day 7 in the roadmap):** turn the model comparison and forecasts into the interactive
# HTML dashboard (Chart.js) — that becomes the presentation layer; this script and its logged
# results are the analytical backbone underneath it, and the source of truth for your report.
