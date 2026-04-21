"""
QQQ three-MA signal combination + statistical validation + multi-feature training
Signal source: QQQ 5/30/200 MA
Trading instrument: TQQQ
"""
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

START = "1999-03-10"
END = "2026-01-17"
INIT_CASH = 10_000.0

print("=" * 100)
print("[Data preparation] QQQ + synthetic TQQQ + VIX + rates + volume")
print("=" * 100)

# Pull data
qqq = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)
qqq_close = qqq["Close"].squeeze()
qqq_vol = qqq["Volume"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tnx = yf.download("^TNX", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()  # 10-year Treasury

# Synthetic TQQQ (corrected accurate version)
def build_synth_tqqq(qqq_close):
    qqq_ret = qqq_close.pct_change().fillna(0)
    expense_daily = 0.0084 / 252
    def fin(d):
        y = d.year
        if y <= 2007: r = 0.045
        elif y <= 2008: r = 0.025
        elif y <= 2015: r = 0.0015
        elif y <= 2019: r = 0.015
        elif y <= 2021: r = 0.001
        else: r = 0.045
        return (r + 0.004) * 2 / 252
    fin_daily = pd.Series([fin(d) for d in qqq_close.index], index=qqq_close.index)
    slip_daily = 0.003 / 252
    synth_ret = 3 * qqq_ret - expense_daily - fin_daily - slip_daily
    return (1 + synth_ret).cumprod()

synth = build_synth_tqqq(qqq_close)
# Calibrate + splice
overlap = tqqq_real.index[0]
calib = float(tqqq_real.iloc[0]) / float(synth.loc[overlap])
synth_pre = synth.loc[:overlap].iloc[:-1] * calib
tqqq_full = pd.concat([synth_pre, tqqq_real])
tqqq_full = tqqq_full.reindex(qqq_close.index).ffill()

print(f"QQQ: {qqq_close.index[0].date()} ~ {qqq_close.index[-1].date()}, {len(qqq_close)} days")
print(f"TQQQ (synthetic+real): start ${tqqq_full.iloc[0]:.4f}, end ${tqqq_full.iloc[-1]:.2f}")
print(f"VIX: {vix.index[0].date()} ~ {vix.index[-1].date()}")
print()

# === Generic backtest function (signal source and trading instrument separable) ===
def backtest(signal, exec_price, fee_bps=2.5, slip_bps=5.0):
    """signal: 0/1 position signal (daily); exec_price: execution price series"""
    pos = signal.shift(1).fillna(0)
    daily_ret = exec_price.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost = pos_change * (fee_bps + slip_bps) / 10000
    ret = pos * daily_ret - cost
    eq = (1 + ret).cumprod() * INIT_CASH
    n_trades = int((pos.diff().abs() > 0).sum())
    return eq, ret, n_trades

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    # Sharpe significance t-statistic
    nobs = len(ret[ret != 0])
    t_stat = sh * np.sqrt(years) if sh else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], t_stat

print("=" * 100)
print("[Experiment 1] QQQ single MA combination -> trade TQQQ - full sample 1999-2026")
print("=" * 100)

ma5 = qqq_close.ewm(span=5, adjust=False).mean()
ma30 = qqq_close.ewm(span=30, adjust=False).mean()
ma200 = qqq_close.ewm(span=200, adjust=False).mean()

# Single crossover signals
sig_5_30 = (ma5 > ma30).astype(int); sig_5_30.iloc[:30] = 0
sig_5_200 = (ma5 > ma200).astype(int); sig_5_200.iloc[:200] = 0
sig_30_200 = (ma30 > ma200).astype(int); sig_30_200.iloc[:200] = 0

# Three-MA joint signal (stacked)
sig_stacked = ((ma5 > ma30) & (ma30 > ma200)).astype(int); sig_stacked.iloc[:200] = 0
# Inverse: if all three MAs trend down -> must be flat, otherwise hold
sig_invert = (~((ma5 < ma30) & (ma30 < ma200))).astype(int); sig_invert.iloc[:200] = 0
# At least 2 of 3 MAs bullish
sig_2of3 = (((ma5 > ma30).astype(int) + (ma5 > ma200).astype(int) + (ma30 > ma200).astype(int)) >= 2).astype(int)
sig_2of3.iloc[:200] = 0

signals = {
    "QQQ 5/30 cross up": sig_5_30,
    "QQQ 5/200 cross up": sig_5_200,
    "QQQ 30/200 cross up (golden cross)": sig_30_200,
    "QQQ 5>30>200 3-layer stacked": sig_stacked,
    "QQQ inverse: not fully crossed down": sig_invert,
    "QQQ 2/3 bullish majority": sig_2of3,
}

print(f"\n{'Signal -> trade TQQQ':<35} {'CAGR':>8} {'MDD':>9} {'Sharpe':>7} {'Calmar':>7} {'Final Value':>14} {'Trades':>5} {'t-stat':>7}")
print("-" * 110)
results = {}
for name, sig in signals.items():
    eq, ret, n = backtest(sig, tqqq_full)
    c, m, s, ca, fv, t = metrics(eq, ret)
    results[name] = (c, m, s, ca, fv, n, t)
    sig_flag = "[sig]" if t > 1.96 else " "
    print(f"{name:<35} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f} {n:>5} {t:>6.2f}{sig_flag}")

print("\n[B&H control]")
bh_eq = (tqqq_full / tqqq_full.iloc[0]) * INIT_CASH
bh_ret = bh_eq.pct_change().fillna(0)
c, m, s, ca, fv, _ = metrics(bh_eq, bh_ret)
print(f"{'TQQQ Buy & Hold':<35} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f} {'-':>5}")

print()
print("=" * 100)
print("[Experiment 2] Strict out-of-sample test: 1999-2010 vs 2010-2026 consistency check")
print("=" * 100)

print(f"\n{'Signal':<35} {'OOS CAGR':>10} {'OOS MDD':>10} {'OOS Calmar':>11} {'IS CAGR':>10} {'IS Calmar':>11} {'Robust?':>7}")
print("-" * 110)
for name, sig in signals.items():
    # 1999-2010
    oos_idx = qqq_close.index <= "2010-02-10"
    eq, ret, _ = backtest(sig[oos_idx], tqqq_full[oos_idx])
    c1, m1, s1, ca1, fv1, _ = metrics(eq, ret)
    # 2010-2026
    is_idx = qqq_close.index >= "2010-02-11"
    eq2, ret2, _ = backtest(sig[is_idx], tqqq_full[is_idx])
    c2, m2, s2, ca2, fv2, _ = metrics(eq2, ret2)
    robust = "YES" if (ca1 > 0 and ca2 > 0) else "NO"
    print(f"{name:<35} {c1*100:>9.2f}% {m1*100:>9.2f}% {ca1:>11.3f} {c2*100:>9.2f}% {ca2:>11.3f} {robust:>7}")

print()
print("=" * 100)
print("[Experiment 3] Walk-forward rolling window (real live-trade simulation)")
print("=" * 100)
print("Rule: every 2 years, use the past 5 years to pick best Calmar from parameter grid, apply for next 2 years")

def run_wf(qqq_close, exec_price, train_years=5, test_years=2):
    """rolling walk-forward, params chosen on train, applied on test"""
    fast_grid = [3, 5, 8, 10, 13, 20]
    slow_grid = [50, 100, 150, 200, 250]

    all_test_returns = []
    test_periods = []
    chosen_params = []

    start_idx = 252 * train_years
    while start_idx + 252 * test_years <= len(qqq_close):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_years, len(qqq_close))
        train_idx = qqq_close.index[train_end - 252*train_years : train_end]
        test_idx = qqq_close.index[train_end : test_end]

        # Select best (by Calmar) on training set
        best_cal = -999
        best_params = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                ema_f = qqq_close.loc[train_idx].ewm(span=f, adjust=False).mean()
                ema_s = qqq_close.loc[train_idx].ewm(span=s, adjust=False).mean()
                sig = (ema_f > ema_s).astype(int); sig.iloc[:s] = 0
                eq, ret, _ = backtest(sig, exec_price.loc[train_idx])
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal
                    best_params = (f, s)

        # Apply best params out-of-sample (test needs train data to compute EMA, so compute together)
        full_idx = qqq_close.index[train_end - 252*train_years : test_end]
        f, s = best_params
        ema_f = qqq_close.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq_close.loc[full_idx].ewm(span=s, adjust=False).mean()
        sig = (ema_f > ema_s).astype(int); sig.iloc[:s] = 0
        sig_test = sig.loc[test_idx]
        exec_test = exec_price.loc[test_idx]
        eq, ret, _ = backtest(sig_test, exec_test)
        all_test_returns.append(ret)
        test_periods.append((test_idx[0], test_idx[-1]))
        chosen_params.append(best_params)
        start_idx += 252 * test_years

    full_ret = pd.concat(all_test_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, test_periods, chosen_params

print("\nRunning (5-year train + 2-year test rolling) ...")
wf_eq, wf_ret, periods, params = run_wf(qqq_close, tqqq_full, 5, 2)
c, m, s, ca, fv, t = metrics(wf_eq, wf_ret)
print(f"\nWalk-forward results (QQQ signal -> TQQQ trading, adaptive params):")
print(f"  CAGR: {c*100:.2f}%, MDD: {m*100:.2f}%, Sharpe: {s:.3f}, Calmar: {ca:.3f}")
print(f"  Final value: ${fv:,.0f}, t-stat: {t:.2f}")
print(f"\nBest parameters chosen per test window:")
for (s_d, e_d), (f, sl) in zip(periods, params):
    print(f"  {s_d.date()} ~ {e_d.date()}: fast={f}, slow={sl}")

print()
print("=" * 100)
print("[Experiment 4] Bootstrap confidence intervals - proper statistical rigor")
print("=" * 100)

def bootstrap_metric(ret, n_boot=1000, block_size=20):
    """Block bootstrap: handles daily return autocorrelation"""
    n = len(ret)
    cagrs = []
    sharpes = []
    np.random.seed(42)
    for _ in range(n_boot):
        # Block bootstrap (block size = 20 days)
        starts = np.random.randint(0, n - block_size, n // block_size)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        idx = idx[idx < n]
        sample_ret = ret.values[idx]
        eq = np.cumprod(1 + sample_ret) * INIT_CASH
        years = len(sample_ret) / 252
        if eq[-1] > 0:
            cagrs.append((eq[-1] / INIT_CASH) ** (1/years) - 1)
        if sample_ret.std() > 0:
            sharpes.append((sample_ret.mean() * 252) / (sample_ret.std() * np.sqrt(252)))
    return np.array(cagrs), np.array(sharpes)

print("\nRun 1000 block bootstraps on real out-of-sample returns from walk-forward:")
cagrs, sharpes = bootstrap_metric(wf_ret, 1000, 20)
print(f"  CAGR  distribution: median {np.median(cagrs)*100:.2f}%, 95% CI [{np.percentile(cagrs, 2.5)*100:.2f}%, {np.percentile(cagrs, 97.5)*100:.2f}%]")
print(f"  Sharpe distribution: median {np.median(sharpes):.3f}, 95% CI [{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]")
print(f"  P(CAGR > 0): {(cagrs > 0).mean()*100:.1f}%")
print(f"  P(Sharpe > 0.5): {(sharpes > 0.5).mean()*100:.1f}%")

print()
print("=" * 100)
print("[Experiment 5] Machine learning signal aggregation - Logistic Regression")
print("=" * 100)
print("Features: QQQ multi-MA + VIX level/change + rates + volume anomaly + historical returns")
print("Label: whether next 5-day TQQQ return > 0")

# Feature engineering
features = pd.DataFrame(index=qqq_close.index)
features['ma_5_30'] = (ma5 / ma30 - 1)
features['ma_5_200'] = (ma5 / ma200 - 1)
features['ma_30_200'] = (ma30 / ma200 - 1)
features['ret_1d'] = qqq_close.pct_change(1)
features['ret_5d'] = qqq_close.pct_change(5)
features['ret_20d'] = qqq_close.pct_change(20)
features['vol_z'] = (qqq_vol - qqq_vol.rolling(20).mean()) / qqq_vol.rolling(20).std()

vix_aligned = vix.reindex(features.index).ffill()
tnx_aligned = tnx.reindex(features.index).ffill()
features['vix'] = vix_aligned
features['vix_change'] = vix_aligned.pct_change(5)
features['tnx'] = tnx_aligned

# Label: whether next 5-day TQQQ return > 0
future_ret = tqqq_full.pct_change(5).shift(-5)
labels = (future_ret > 0).astype(int)

# Align + drop NaN
data = pd.concat([features, labels.rename('y')], axis=1).dropna()
X_full = data.drop(columns=['y'])
y_full = data['y']

# Walk-forward train LR
print("\nWalk-forward LR (5-year train -> 2-year test):")
all_preds = []
test_idx_all = []
years_per_test = 2
years_per_train = 5
days_per_year = 252

start = years_per_train * days_per_year
while start + years_per_test * days_per_year <= len(X_full):
    train_X = X_full.iloc[start - years_per_train * days_per_year : start]
    train_y = y_full.iloc[start - years_per_train * days_per_year : start]
    test_X = X_full.iloc[start : start + years_per_test * days_per_year]

    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    test_Xs = scaler.transform(test_X)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(train_Xs, train_y)
    pred_proba = model.predict_proba(test_Xs)[:, 1]

    all_preds.append(pd.Series(pred_proba, index=test_X.index))
    test_idx_all.append(test_X.index)
    start += years_per_test * days_per_year

all_preds_series = pd.concat(all_preds)
# Treat proba > 0.55 as long signal
ml_signal = (all_preds_series > 0.55).astype(int).reindex(qqq_close.index).fillna(0)
exec_aligned = tqqq_full.loc[all_preds_series.index]
eq_ml, ret_ml, n_ml = backtest(ml_signal.loc[all_preds_series.index], exec_aligned)
c, m, s, ca, fv, t = metrics(eq_ml, ret_ml)
print(f"  LR walk-forward: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {s:.3f}, Calmar {ca:.3f}, Trades {n_ml}")

print()
print("=" * 100)
print("[Experiment 6] Random Forest + feature importance")
print("=" * 100)
all_preds_rf = []
all_importance = []
start = years_per_train * days_per_year
while start + years_per_test * days_per_year <= len(X_full):
    train_X = X_full.iloc[start - years_per_train * days_per_year : start]
    train_y = y_full.iloc[start - years_per_train * days_per_year : start]
    test_X = X_full.iloc[start : start + years_per_test * days_per_year]

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(train_X, train_y)
    pred_proba = rf.predict_proba(test_X)[:, 1]
    all_preds_rf.append(pd.Series(pred_proba, index=test_X.index))
    all_importance.append(rf.feature_importances_)
    start += years_per_test * days_per_year

all_preds_rf = pd.concat(all_preds_rf)
rf_signal = (all_preds_rf > 0.55).astype(int).reindex(qqq_close.index).fillna(0)
exec_aligned = tqqq_full.loc[all_preds_rf.index]
eq_rf, ret_rf, n_rf = backtest(rf_signal.loc[all_preds_rf.index], exec_aligned)
c, m, s, ca, fv, t = metrics(eq_rf, ret_rf)
print(f"  RF walk-forward: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {s:.3f}, Calmar {ca:.3f}, Trades {n_rf}")

# Average feature importance
avg_imp = np.mean(all_importance, axis=0)
imp_df = pd.DataFrame({"feature": X_full.columns, "importance": avg_imp}).sort_values("importance", ascending=False)
print(f"\nFeature importance (RF average):")
for _, row in imp_df.iterrows():
    print(f"  {row['feature']:<15} {row['importance']*100:.2f}%")

print()
print("=" * 100)
print("[Experiment 7] Final comparison - all methods' real out-of-sample performance")
print("=" * 100)
# Walk-forward comparison of all methods
print(f"\n{'Method':<40} {'CAGR':>9} {'MDD':>9} {'Sharpe':>7} {'Calmar':>7} {'Final Value':>14}")
print("-" * 100)

# WF EMA
c, m, s, ca, fv, _ = metrics(wf_eq, wf_ret)
print(f"{'Walk-forward EMA (adaptive params)':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

# WF LR
c, m, s, ca, fv, _ = metrics(eq_ml, ret_ml)
print(f"{'Walk-forward Logistic Regression':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

# WF RF
c, m, s, ca, fv, _ = metrics(eq_rf, ret_rf)
print(f"{'Walk-forward Random Forest':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

# B&H control (same period)
test_period = ret_ml.index
bh_test = tqqq_full.loc[test_period]
bh_eq_t = (bh_test / bh_test.iloc[0]) * INIT_CASH
bh_ret_t = bh_eq_t.pct_change().fillna(0)
c, m, s, ca, fv, _ = metrics(bh_eq_t, bh_ret_t)
print(f"{'TQQQ B&H (same period)':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

bh_qqq = qqq_close.loc[test_period]
bh_qqq_eq = (bh_qqq / bh_qqq.iloc[0]) * INIT_CASH
bh_qqq_ret = bh_qqq_eq.pct_change().fillna(0)
c, m, s, ca, fv, _ = metrics(bh_qqq_eq, bh_qqq_ret)
print(f"{'QQQ B&H (same period)':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")
