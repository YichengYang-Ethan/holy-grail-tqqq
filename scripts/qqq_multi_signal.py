"""
QQQ 三 MA 信号组合 + 统计学验证 + 多特征训练
信号源: QQQ 5/30/200 MA
交易标的: TQQQ
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
print("【数据准备】QQQ + 合成 TQQQ + VIX + 利率 + 成交量")
print("=" * 100)

# 拉数据
qqq = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)
qqq_close = qqq["Close"].squeeze()
qqq_vol = qqq["Volume"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tnx = yf.download("^TNX", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()  # 10年期国债

# 合成 TQQQ (修正后的精确版)
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
# 校准 + 拼接
overlap = tqqq_real.index[0]
calib = float(tqqq_real.iloc[0]) / float(synth.loc[overlap])
synth_pre = synth.loc[:overlap].iloc[:-1] * calib
tqqq_full = pd.concat([synth_pre, tqqq_real])
tqqq_full = tqqq_full.reindex(qqq_close.index).ffill()

print(f"QQQ: {qqq_close.index[0].date()} ~ {qqq_close.index[-1].date()}, {len(qqq_close)} 天")
print(f"TQQQ (合成+真实): 起 ${tqqq_full.iloc[0]:.4f}, 终 ${tqqq_full.iloc[-1]:.2f}")
print(f"VIX: {vix.index[0].date()} ~ {vix.index[-1].date()}")
print()

# === 通用回测函数（信号源 vs 交易标的可分离）===
def backtest(signal, exec_price, fee_bps=2.5, slip_bps=5.0):
    """signal: 0/1 持仓信号 (按日)；exec_price: 交易价格序列"""
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
    # Sharpe 显著性 t-statistic
    nobs = len(ret[ret != 0])
    t_stat = sh * np.sqrt(years) if sh else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], t_stat

print("=" * 100)
print("【实验 1】QQQ 单 MA 组合 → 交易 TQQQ — 全样本 1999-2026")
print("=" * 100)

ma5 = qqq_close.ewm(span=5, adjust=False).mean()
ma30 = qqq_close.ewm(span=30, adjust=False).mean()
ma200 = qqq_close.ewm(span=200, adjust=False).mean()

# 单交叉信号
sig_5_30 = (ma5 > ma30).astype(int); sig_5_30.iloc[:30] = 0
sig_5_200 = (ma5 > ma200).astype(int); sig_5_200.iloc[:200] = 0
sig_30_200 = (ma30 > ma200).astype(int); sig_30_200.iloc[:200] = 0

# 三 MA 联立信号 (stacked)
sig_stacked = ((ma5 > ma30) & (ma30 > ma200)).astype(int); sig_stacked.iloc[:200] = 0
# 反向：三 MA 都向下 → 必须空仓，否则持仓
sig_invert = (~((ma5 < ma30) & (ma30 < ma200))).astype(int); sig_invert.iloc[:200] = 0
# 至少 2 个 MA 看多
sig_2of3 = (((ma5 > ma30).astype(int) + (ma5 > ma200).astype(int) + (ma30 > ma200).astype(int)) >= 2).astype(int)
sig_2of3.iloc[:200] = 0

signals = {
    "QQQ 5/30 上穿": sig_5_30,
    "QQQ 5/200 上穿": sig_5_200,
    "QQQ 30/200 上穿 (黄金交叉)": sig_30_200,
    "QQQ 5>30>200 三层堆叠": sig_stacked,
    "QQQ 反向: 非全死叉": sig_invert,
    "QQQ 2/3 看多多数票": sig_2of3,
}

print(f"\n{'信号 → 交易 TQQQ':<35} {'CAGR':>8} {'MDD':>9} {'夏普':>7} {'卡玛':>7} {'终值':>14} {'交易':>5} {'t-stat':>7}")
print("-" * 110)
results = {}
for name, sig in signals.items():
    eq, ret, n = backtest(sig, tqqq_full)
    c, m, s, ca, fv, t = metrics(eq, ret)
    results[name] = (c, m, s, ca, fv, n, t)
    sig_flag = "✅" if t > 1.96 else " "
    print(f"{name:<35} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f} {n:>5} {t:>6.2f}{sig_flag}")

print("\n[B&H 对照]")
bh_eq = (tqqq_full / tqqq_full.iloc[0]) * INIT_CASH
bh_ret = bh_eq.pct_change().fillna(0)
c, m, s, ca, fv, _ = metrics(bh_eq, bh_ret)
print(f"{'TQQQ Buy & Hold':<35} {c*100:>7.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f} {'-':>5}")

print()
print("=" * 100)
print("【实验 2】样本外严格测试：1999-2010 vs 2010-2026 一致性检验")
print("=" * 100)

print(f"\n{'信号':<35} {'OOS CAGR':>10} {'OOS MDD':>10} {'OOS Calmar':>11} {'IS CAGR':>10} {'IS Calmar':>11} {'稳健?':>7}")
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
    robust = "✅" if (ca1 > 0 and ca2 > 0) else "❌"
    print(f"{name:<35} {c1*100:>9.2f}% {m1*100:>9.2f}% {ca1:>11.3f} {c2*100:>9.2f}% {ca2:>11.3f} {robust:>7}")

print()
print("=" * 100)
print("【实验 3】Walk-Forward 滚动窗口（真正模拟实盘可行）")
print("=" * 100)
print("规则：每 2 年用过去 5 年数据从参数网格选最优 Calmar，下 2 年用之）")

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

        # 在训练集上选最优 (按 Calmar)
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

        # 用最优参数测样本外 (test 期需要 train 期的数据来计算 EMA, 所以一起算)
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

print("\n运行中（5年训练 + 2年测试 滚动）...")
wf_eq, wf_ret, periods, params = run_wf(qqq_close, tqqq_full, 5, 2)
c, m, s, ca, fv, t = metrics(wf_eq, wf_ret)
print(f"\nWalk-forward 结果（QQQ 信号 → TQQQ 交易，自适应参数）:")
print(f"  CAGR: {c*100:.2f}%, MDD: {m*100:.2f}%, 夏普: {s:.3f}, 卡玛: {ca:.3f}")
print(f"  终值: ${fv:,.0f}, t-stat: {t:.2f}")
print(f"\n各 test 窗口选出的最优参数:")
for (s_d, e_d), (f, sl) in zip(periods, params):
    print(f"  {s_d.date()} ~ {e_d.date()}: fast={f}, slow={sl}")

print()
print("=" * 100)
print("【实验 4】Bootstrap 置信区间 — 真正的统计学严谨")
print("=" * 100)

def bootstrap_metric(ret, n_boot=1000, block_size=20):
    """Block bootstrap: 处理日收益自相关性"""
    n = len(ret)
    cagrs = []
    sharpes = []
    np.random.seed(42)
    for _ in range(n_boot):
        # Block bootstrap (block size = 20 day)
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

print("\n用 walk-forward 出来的真实样本外收益做 1000 次 block bootstrap:")
cagrs, sharpes = bootstrap_metric(wf_ret, 1000, 20)
print(f"  CAGR  分布: 中位数 {np.median(cagrs)*100:.2f}%, 95% CI [{np.percentile(cagrs, 2.5)*100:.2f}%, {np.percentile(cagrs, 97.5)*100:.2f}%]")
print(f"  夏普  分布: 中位数 {np.median(sharpes):.3f}, 95% CI [{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]")
print(f"  CAGR > 0 概率: {(cagrs > 0).mean()*100:.1f}%")
print(f"  夏普 > 0.5 概率: {(sharpes > 0.5).mean()*100:.1f}%")

print()
print("=" * 100)
print("【实验 5】机器学习信号聚合 - Logistic Regression")
print("=" * 100)
print("特征: QQQ 多 MA + VIX 水平/变化 + 利率 + 成交量异常 + 历史回报")
print("标签: 未来 5 日 TQQQ 收益是否 > 0")

# 特征工程
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

# 标签：未来 5 天 TQQQ 收益是否 > 0
future_ret = tqqq_full.pct_change(5).shift(-5)
labels = (future_ret > 0).astype(int)

# 对齐 + 去 NaN
data = pd.concat([features, labels.rename('y')], axis=1).dropna()
X_full = data.drop(columns=['y'])
y_full = data['y']

# Walk-forward 训练 LR
print("\nWalk-forward LR (5年训练 → 2年测试):")
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
# 高于 0.55 视为做多信号
ml_signal = (all_preds_series > 0.55).astype(int).reindex(qqq_close.index).fillna(0)
exec_aligned = tqqq_full.loc[all_preds_series.index]
eq_ml, ret_ml, n_ml = backtest(ml_signal.loc[all_preds_series.index], exec_aligned)
c, m, s, ca, fv, t = metrics(eq_ml, ret_ml)
print(f"  LR walk-forward: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, 夏普 {s:.3f}, 卡玛 {ca:.3f}, 交易 {n_ml}")

print()
print("=" * 100)
print("【实验 6】Random Forest + 特征重要性")
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
print(f"  RF walk-forward: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, 夏普 {s:.3f}, 卡玛 {ca:.3f}, 交易 {n_rf}")

# 平均特征重要性
avg_imp = np.mean(all_importance, axis=0)
imp_df = pd.DataFrame({"feature": X_full.columns, "importance": avg_imp}).sort_values("importance", ascending=False)
print(f"\n特征重要性 (RF 平均):")
for _, row in imp_df.iterrows():
    print(f"  {row['feature']:<15} {row['importance']*100:.2f}%")

print()
print("=" * 100)
print("【实验 7】最终对比 — 所有方法在严格样本外的真实表现")
print("=" * 100)
# 所有方法的 walk-forward 结果对比
print(f"\n{'方法':<40} {'CAGR':>9} {'MDD':>9} {'夏普':>7} {'卡玛':>7} {'终值':>14}")
print("-" * 100)

# WF EMA
c, m, s, ca, fv, _ = metrics(wf_eq, wf_ret)
print(f"{'Walk-forward EMA (自适应参数)':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

# WF LR
c, m, s, ca, fv, _ = metrics(eq_ml, ret_ml)
print(f"{'Walk-forward Logistic Regression':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

# WF RF
c, m, s, ca, fv, _ = metrics(eq_rf, ret_rf)
print(f"{'Walk-forward Random Forest':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

# B&H 对照（同期）
test_period = ret_ml.index
bh_test = tqqq_full.loc[test_period]
bh_eq_t = (bh_test / bh_test.iloc[0]) * INIT_CASH
bh_ret_t = bh_eq_t.pct_change().fillna(0)
c, m, s, ca, fv, _ = metrics(bh_eq_t, bh_ret_t)
print(f"{'TQQQ B&H (同时段)':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")

bh_qqq = qqq_close.loc[test_period]
bh_qqq_eq = (bh_qqq / bh_qqq.iloc[0]) * INIT_CASH
bh_qqq_ret = bh_qqq_eq.pct_change().fillna(0)
c, m, s, ca, fv, _ = metrics(bh_qqq_eq, bh_qqq_ret)
print(f"{'QQQ B&H (同时段)':<40} {c*100:>8.2f}% {m*100:>8.2f}% {s:>7.3f} {ca:>7.3f} {fv:>14,.0f}")
