"""
Testing classic low-drawdown strategies for MDD < 20% target:
- 60/40 stocks/bonds (SPY/AGG)
- Permanent Portfolio (25% SPY + 25% TLT + 25% GLD + 25% BIL)
- Risk Parity (inverse-vol weighting SPY + TLT + GLD)
- Vol-targeted QQQ (dynamic leverage to target 10% annualized vol)
- 40/60 stocks/bonds (conservative)
- 30/70 stocks/bonds (bond-heavy)

Tested via CPCV (45 paths) on 1999-2026 where data exists.
Compare to Rotation winner.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

INIT = 10_000.0
EMBARGO = 21

# Data
print("Loading multi-asset data ...")
START = "1999-03-10"
END = "2026-04-18"

tickers = {
    "SPY": "SPY",        # S&P 500
    "AGG": "AGG",        # US aggregate bonds
    "TLT": "TLT",        # 20+ year treasuries
    "GLD": "GLD",        # Gold
    "BIL": "BIL",        # 1-3 month T-bills
    "QQQ": "QQQ",
}

# Download, using earliest available for each
data = {}
for name, sym in tickers.items():
    d = yf.download(sym, start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
    data[name] = d
    print(f"  {name}: {d.index[0].date()} ~ {d.index[-1].date()}, {len(d)} days")

# Use common date range (the latest start)
common_start = max(d.index[0] for d in data.values())
print(f"\nCommon start: {common_start.date()}")

# Align everything to common dates
idx = None
for name, d in data.items():
    d = d.loc[common_start:]
    if idx is None: idx = d.index
    else: idx = idx.intersection(d.index)

aligned = {name: data[name].loc[idx] for name in data}
print(f"Aligned: {idx[0].date()} ~ {idx[-1].date()}, {len(idx)} days")

def metrics(eq, ret):
    if eq.empty or eq.iloc[-1] <= 0:
        return -1, -1, 0, -1, 0, 0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0:
        return 0, 0, 0, 0, 0, 0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else cagr
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

def portfolio_backtest(weights, prices_dict, rebalance_days=63, fee_bps=2.5, slip_bps=5.0):
    """
    Weighted portfolio with periodic rebalance.
    weights: {asset: weight} summing to 1
    prices_dict: {asset: price series}
    rebalance_days: 63 = quarterly
    """
    common = list(prices_dict.values())[0].index
    # Compute each asset's daily returns
    rets = pd.DataFrame({a: p.pct_change().fillna(0) for a, p in prices_dict.items()}, index=common)
    # Build weight series that rebalances periodically
    n = len(common)
    port_val = INIT
    port_hist = pd.Series(index=common, dtype=float)
    current_weights = {a: weights[a] for a in weights}
    last_rebal = 0
    for i, t in enumerate(common):
        # Grow each asset position
        if i > 0:
            for a in current_weights:
                current_weights[a] = current_weights[a] * (1 + rets[a].iloc[i])
            total = sum(current_weights.values())
            if total > 0:
                # normalize to current portfolio value fraction
                pass
        # Rebalance?
        if i - last_rebal >= rebalance_days and i > 0:
            total = sum(current_weights.values())
            # Turnover cost
            turnover = sum(abs(current_weights[a]/total - weights[a]) for a in weights)
            cost = total * turnover * (fee_bps + slip_bps) / 10000
            total -= cost
            current_weights = {a: weights[a] * total for a in weights}
            last_rebal = i
        port_val = sum(current_weights.values())
        port_hist.iloc[i] = port_val
    # Normalize to start at INIT
    port_hist = port_hist / port_hist.iloc[0] * INIT
    strat_ret = port_hist.pct_change().fillna(0)
    return port_hist, strat_ret

# Strategies
STRATEGIES = {
    "60/40 SPY/AGG": {"SPY": 0.60, "AGG": 0.40},
    "50/50 SPY/AGG": {"SPY": 0.50, "AGG": 0.50},
    "40/60 SPY/AGG": {"SPY": 0.40, "AGG": 0.60},
    "30/70 SPY/AGG": {"SPY": 0.30, "AGG": 0.70},
    "Permanent Portfolio": {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "BIL": 0.25},
    "60/40 QQQ/AGG": {"QQQ": 0.60, "AGG": 0.40},
    "50/50 QQQ/TLT": {"QQQ": 0.50, "TLT": 0.50},
}

# Full-period backtest
print(f"\n{'=' * 110}")
print("SINGLE-PATH FULL HISTORY")
print("=" * 110)
print(f"\n{'Strategy':<30} {'CAGR':>8} {'MDD':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Final $10K':>14}")
print("-" * 110)

single_path = {}
for name, weights in STRATEGIES.items():
    prices_needed = {a: aligned[a] for a in weights}
    eq, ret = portfolio_backtest(weights, prices_needed)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    single_path[name] = {"eq": eq, "ret": ret, "metrics": (c, m, sh, ca, fv, so)}
    mdd_flag = "✅" if m > -0.20 else ("⚠️" if m > -0.30 else "❌")
    print(f"{name:<30} {c*100:>7.2f}% {m*100:>8.2f}% {sh:>8.3f} {so:>9.3f} {ca:>8.3f} {fv:>14,.0f}  {mdd_flag}")

# CPCV
print(f"\n{'=' * 110}")
print(f"CPCV (N=10, k=2 → 45 paths)")
print("=" * 110)

def cpcv_portfolio(weights, prices_dict, n_splits=10, k_test=2, embargo=21, rebalance_days=63):
    common = list(prices_dict.values())[0].index
    n = len(common)
    fold_size = n // n_splits
    folds = [(i * fold_size, (i+1) * fold_size if i < n_splits-1 else n) for i in range(n_splits)]
    all_combos = list(combinations(range(n_splits), k_test))
    path_results = []
    for combo in all_combos:
        test_returns = []
        for tf in combo:
            tf_start, tf_end = folds[tf]
            tf_start_emb = tf_start + embargo
            tf_end_emb = max(tf_start_emb + 1, tf_end - embargo)
            if tf_end_emb - tf_start_emb < 100: continue
            test_idx = common[tf_start_emb:tf_end_emb]
            # Slice prices to test period
            sliced = {a: prices_dict[a].loc[test_idx] for a in prices_dict}
            eq, ret = portfolio_backtest(weights, sliced, rebalance_days)
            if not ret.empty:
                test_returns.append(ret)
        if test_returns:
            combined = pd.concat(test_returns)
            combined_eq = (1 + combined).cumprod() * INIT
            c, m, sh, ca, fv, so = metrics(combined_eq, combined)
            path_results.append({'cagr': c, 'mdd': m, 'sharpe': sh, 'calmar': ca})
    return path_results

cpcv_results = {}
for name, weights in STRATEGIES.items():
    prices_needed = {a: aligned[a] for a in weights}
    r = cpcv_portfolio(weights, prices_needed, rebalance_days=63)
    cpcv_results[name] = r

print(f"\n{'Strategy':<30} {'Med CAGR':>10} {'CAGR IQR':>22} {'Med Sharpe':>11} {'Med Calmar':>11} {'Med MDD':>10} {'P(>0)':>7}")
print("-" * 120)
for name, results in cpcv_results.items():
    if not results: continue
    cagrs = np.array([p['cagr'] for p in results])
    sharpes = np.array([p['sharpe'] for p in results])
    calmars = np.array([p['calmar'] for p in results])
    mdds = np.array([p['mdd'] for p in results])
    q25 = np.percentile(cagrs, 25) * 100
    q75 = np.percentile(cagrs, 75) * 100
    iqr = f"[{q25:>6.2f}%, {q75:>6.2f}%]"
    print(f"{name:<30} {np.median(cagrs)*100:>9.2f}% {iqr:>22} {np.median(sharpes):>11.3f} {np.median(calmars):>11.3f} {np.median(mdds)*100:>9.2f}% {(cagrs>0).mean()*100:>6.1f}%")

# Compare to Rotation
print(f"\n{'=' * 110}")
print("COMPARISON TABLE (LOW-DD OPTIONS):")
print("=" * 110)
print(f"\nTarget: MDD < 20% while maintaining meaningful CAGR")
print(f"\nSingle-path ranking by Calmar:")

ranked = sorted(single_path.items(), key=lambda x: -x[1]['metrics'][3])
for name, d in ranked:
    c, m, sh, ca, fv, so = d['metrics']
    print(f"  {name:<30} CAGR {c*100:>6.2f}%  MDD {m*100:>7.2f}%  Calmar {ca:.3f}")
