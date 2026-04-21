"""
纯 TQQQ 金字塔策略（无 QQQ）
目标: 找到合理 / 不过拟合的 TQQQ + 现金金字塔, 击败 WF MA5/200 TQQQ rotation
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0

print("[1/6] 数据准备 ...")
qqq = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()

def build_synth(qqq_close, tqqq_real):
    qret = qqq_close.pct_change().fillna(0)
    expense_d = 0.0084 / 252
    def fin(d):
        y = d.year
        if y <= 2007: r = 0.045
        elif y <= 2008: r = 0.025
        elif y <= 2015: r = 0.0015
        elif y <= 2019: r = 0.015
        elif y <= 2021: r = 0.001
        else: r = 0.045
        return (r + 0.004) * 2 / 252
    fin_d = pd.Series([fin(d) for d in qqq_close.index], index=qqq_close.index)
    slip_d = 0.003 / 252
    synth_ret = 3 * qret - expense_d - fin_d - slip_d
    synth = (1 + synth_ret).cumprod()
    overlap = tqqq_real.index[0]
    calib = float(tqqq_real.iloc[0]) / float(synth.loc[overlap])
    pre = synth.loc[:overlap].iloc[:-1] * calib
    full = pd.concat([pre, tqqq_real])
    return full.reindex(qqq_close.index).ffill()

tqqq_full = build_synth(qqq, tqqq_real)
print(f"  TQQQ {tqqq_full.index[0].date()} ~ {tqqq_full.index[-1].date()}, {len(tqqq_full)} 天")
print()

def backtest_tqqq_cash(tpos, tqqq, fee_bps=2.5, slip_bps=5.0):
    """TQQQ + 现金。tpos = TQQQ 权重, 1-tpos = 现金"""
    tpos_lag = tpos.shift(1).fillna(0)
    tret = tqqq.pct_change().fillna(0)
    pos_change = tpos_lag.diff().abs().fillna(0)
    cost = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = tpos_lag * tret - cost
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    n_trades = int((pos_change > 0.05).sum())
    return eq, strat_ret, n_trades

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    so = (ret.mean() * 252) / (ret[ret<0].std() * np.sqrt(252)) if ret[ret<0].std() > 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1], so

# ============================================================
# 信号源: 牛/熊判断 (用 EMA 5/200 作为基础)
# ============================================================
def get_bull_signal(qqq, fast=5, slow=200):
    ema_f = qqq.ewm(span=fast, adjust=False).mean()
    ema_s = qqq.ewm(span=slow, adjust=False).mean()
    bull = (ema_f > ema_s)
    bull.iloc[:slow] = False
    return bull

# ============================================================
# 金字塔策略生成函数
# ============================================================
def build_pure_tqqq_pyramid(qqq, tqqq, bull_signal, ma_period=200,
                              bear_base_tqqq=0.0,  # 熊市基础 TQQQ 持仓
                              levels=None,  # [(drawdown, deploy_fraction)]
                              anchor="ma"):  # "ma" 或 "peak"
    """
    纯 TQQQ + 现金的金字塔策略
    - bear_base_tqqq: 熊市初始 TQQQ 仓位 (0 = 全现金, 0.5 = 半仓)
    - levels: 金字塔阶梯 [(跌幅, 累计部署比例)]
    - anchor: "ma" 用 MA 偏离, "peak" 用最高点回撤
    """
    if levels is None:
        levels = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]

    if anchor == "ma":
        ma = qqq.ewm(span=ma_period, adjust=False).mean()
        deviation = (qqq / ma - 1)
    else:  # peak
        peak = qqq.cummax()
        deviation = (qqq / peak - 1)

    tpos = pd.Series(0.0, index=qqq.index)

    for i in range(len(qqq.index)):
        if i < ma_period:
            tpos.iloc[i] = 0
            continue

        if bull_signal.iloc[i]:
            tpos.iloc[i] = 1.0
        else:
            # 熊市: 计算金字塔部署
            dev = deviation.iloc[i]
            deploy = 0.0
            for thresh, frac in levels:
                if dev <= thresh:
                    deploy = frac
            cash_reserve = 1.0 - bear_base_tqqq
            tpos.iloc[i] = bear_base_tqqq + cash_reserve * deploy

    return tpos

# ============================================================
# 实验 1: 多种纯 TQQQ 金字塔配置 - 全样本
# ============================================================
print("=" * 100)
print("【实验 1】纯 TQQQ + 现金金字塔配置扫描 — 全样本 1999-2026")
print("=" * 100)

bull = get_bull_signal(qqq, 5, 200)

configs = [
    # (name, base_tqqq, levels, anchor, ma_period)
    ("T1 全现金/MA200锚 -10/-20/-30/-40 → 25/50/75/100", 0.0, [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)], "ma", 200),
    ("T2 全现金/MA200锚 -5/-15/-25/-35 → 25/50/75/100",  0.0, [(-0.05,0.25),(-0.15,0.50),(-0.25,0.75),(-0.35,1.00)], "ma", 200),
    ("T3 全现金/MA200锚 -15/-25/-35/-45 → 25/50/75/100", 0.0, [(-0.15,0.25),(-0.25,0.50),(-0.35,0.75),(-0.45,1.00)], "ma", 200),
    ("T4 全现金/Peak锚 -10/-20/-30/-40 → 25/50/75/100",  0.0, [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)], "peak", 200),
    ("T5 全现金/Peak锚 -15/-30/-45/-60 → 25/50/75/100",  0.0, [(-0.15,0.25),(-0.30,0.50),(-0.45,0.75),(-0.60,1.00)], "peak", 200),
    ("T6 半仓/MA200锚 -10/-20/-30/-40 → 25/50/75/100",   0.5, [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)], "ma", 200),
    ("T7 30%基础/MA200锚 -10/-20/-30/-40",              0.3, [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)], "ma", 200),
    ("T8 全现金/MA50锚 -5/-10/-15/-20",                  0.0, [(-0.05,0.25),(-0.10,0.50),(-0.15,0.75),(-0.20,1.00)], "ma", 50),
    ("T9 全现金/MA200锚 多档 -5/-10/-20/-30/-40/-50",      0.0, [(-0.05,0.20),(-0.10,0.40),(-0.20,0.60),(-0.30,0.80),(-0.40,1.00)], "ma", 200),
    ("T10 全现金/MA200锚 简单 -20/-40 → 50/100",          0.0, [(-0.20,0.50),(-0.40,1.00)], "ma", 200),
    ("T11 全现金/MA200锚 极早 0/-10/-20/-30",             0.0, [(0.0,0.25),(-0.10,0.50),(-0.20,0.75),(-0.30,1.00)], "ma", 200),
]

print(f"\n{'配置':<60} {'CAGR':>7} {'MDD':>9} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'终值':>14}")
print("-" * 130)
results_full = {}
for name, base, levels, anchor, ma_p in configs:
    tpos = build_pure_tqqq_pyramid(qqq, tqqq_full, bull, ma_p, base, levels, anchor)
    eq, ret, n = backtest_tqqq_cash(tpos, tqqq_full)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    results_full[name] = (c, m, sh, ca, fv, so)
    print(f"{name:<60} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# 基准: WF MA5/200 TQQQ rotation (cash 版本)
print("\n基准对比:")
print("-" * 130)

def wf_tqqq_cash(qqq, tqqq, train_y=5, test_y=2):
    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    start_idx = 252 * train_y
    while start_idx + 252 * test_y <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_y, len(qqq))
        train_idx = qqq.index[train_end - 252*train_y : train_end]
        test_idx = qqq.index[train_end : test_end]
        best_cal = -999
        best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                ema_f = qqq.loc[train_idx].ewm(span=f, adjust=False).mean()
                ema_s = qqq.loc[train_idx].ewm(span=s, adjust=False).mean()
                bull_t = (ema_f > ema_s).astype(float); bull_t.iloc[:s] = 0
                eq, ret, _ = backtest_tqqq_cash(bull_t, tqqq.loc[train_idx])
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_y : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        bull_t = (ema_f > ema_s).astype(float); bull_t.iloc[:s] = 0
        bull_test = bull_t.loc[test_idx]
        eq, ret, _ = backtest_tqqq_cash(bull_test, tqqq.loc[test_idx])
        all_returns.append(ret)
        start_idx += 252 * test_y
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret

# 单一参数 MA5/200 (无 WF) 全样本
tpos_basic = bull.astype(float)
eq, ret, _ = backtest_tqqq_cash(tpos_basic, tqqq_full)
c, m, sh, ca, fv, so = metrics(eq, ret)
print(f"{'★ 基准 1: 静态 EMA5/200 TQQQ/cash':<60} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# B&H
eq = (tqqq_full / tqqq_full.iloc[0]) * INIT_CASH
ret = eq.pct_change().fillna(0)
c, m, sh, ca, fv, so = metrics(eq, ret)
print(f"{'★ 基准 2: TQQQ Buy & Hold':<60} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

eq = (qqq / qqq.iloc[0]) * INIT_CASH
ret = eq.pct_change().fillna(0)
c, m, sh, ca, fv, so = metrics(eq, ret)
print(f"{'★ 基准 3: QQQ Buy & Hold':<60} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

print()
print("=" * 100)
print("【实验 2】Walk-Forward 纯 TQQQ + 金字塔（参数自适应）")
print("=" * 100)

def wf_pure_tqqq_pyramid(qqq, tqqq, levels, base_tqqq=0.0, anchor="ma",
                         train_y=5, test_y=2):
    """walk-forward: 只调 (fast, slow), 金字塔参数固定"""
    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    chosen = []
    start_idx = 252 * train_y
    while start_idx + 252 * test_y <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_y, len(qqq))
        train_idx = qqq.index[train_end - 252*train_y : train_end]
        test_idx = qqq.index[train_end : test_end]

        best_cal = -999; best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                bull_t = get_bull_signal(qqq.loc[train_idx], f, s)
                tpos = build_pure_tqqq_pyramid(qqq.loc[train_idx], tqqq.loc[train_idx], bull_t, s, base_tqqq, levels, anchor)
                eq, ret, _ = backtest_tqqq_cash(tpos, tqqq.loc[train_idx])
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)

        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_y : test_end]
        bull_t = get_bull_signal(qqq.loc[full_idx], f, s)
        tpos = build_pure_tqqq_pyramid(qqq.loc[full_idx], tqqq.loc[full_idx], bull_t, s, base_tqqq, levels, anchor)
        tpos_test = tpos.loc[test_idx]
        eq, ret, _ = backtest_tqqq_cash(tpos_test, tqqq.loc[test_idx])
        all_returns.append(ret)
        chosen.append(best_p)
        start_idx += 252 * test_y
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen

print("\n运行 WF 纯 TQQQ + 多种金字塔配置 (5y train + 2y test) ...")
wf_configs = [
    ("WF-T1 全现金/MA200锚 -10/-20/-30/-40", [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)], 0.0, "ma"),
    ("WF-T2 全现金/MA200锚 -5/-15/-25/-35",  [(-0.05,0.25),(-0.15,0.50),(-0.25,0.75),(-0.35,1.00)], 0.0, "ma"),
    ("WF-T4 全现金/Peak锚 -10/-20/-30/-40",  [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)], 0.0, "peak"),
    ("WF-T5 全现金/Peak锚 -15/-30/-45/-60",  [(-0.15,0.25),(-0.30,0.50),(-0.45,0.75),(-0.60,1.00)], 0.0, "peak"),
    ("WF-T6 半仓/MA200锚 -10/-20/-30/-40",   [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)], 0.5, "ma"),
    ("WF-T7 30%基础/MA200锚 -10/-20/-30/-40", [(-0.10,0.25),(-0.20,0.50),(-0.30,0.75),(-0.40,1.00)], 0.3, "ma"),
    ("WF-T11 极早 0/-10/-20/-30",            [(0.0,0.25),(-0.10,0.50),(-0.20,0.75),(-0.30,1.00)], 0.0, "ma"),
]

print(f"\n{'WF 配置':<58} {'CAGR':>7} {'MDD':>9} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'终值':>14}")
print("-" * 130)
wf_results = {}
for name, levels, base, anchor in wf_configs:
    eq, ret, params = wf_pure_tqqq_pyramid(qqq, tqqq_full, levels, base, anchor, 5, 2)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    wf_results[name] = (c, m, sh, ca, fv, so, eq, ret)
    print(f"{name:<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# 基准: WF cash 版 (无金字塔)
print("\n--- 基准 ---")
print("-" * 130)
eq, ret = wf_tqqq_cash(qqq, tqqq_full, 5, 2)
c, m, sh, ca, fv, so = metrics(eq, ret)
print(f"{'★ WF MA5/200 TQQQ/cash 纯轮换 (cash 版)':<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# WF rotation (QQQ/TQQQ) 同期
def wf_qqq_tqqq(qqq, tqqq_full, train_y=5, test_y=2):
    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    start_idx = 252 * train_y
    while start_idx + 252 * test_y <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_y, len(qqq))
        train_idx = qqq.index[train_end - 252*train_y : train_end]
        test_idx = qqq.index[train_end : test_end]
        best_cal = -999; best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                bull_t = get_bull_signal(qqq.loc[train_idx], f, s).astype(float)
                # rotation: bull → TQQQ, bear → QQQ
                qpos = (1 - bull_t)
                tpos = bull_t
                qret = qqq.loc[train_idx].pct_change().fillna(0)
                tret = tqqq_full.loc[train_idx].pct_change().fillna(0)
                qpos_lag = qpos.shift(1).fillna(0); tpos_lag = tpos.shift(1).fillna(0)
                cost = (qpos_lag.diff().abs().fillna(0) + tpos_lag.diff().abs().fillna(0)) * 7.5/10000
                strat_ret = qpos_lag * qret + tpos_lag * tret - cost
                eq = (1 + strat_ret).cumprod() * INIT_CASH
                _, _, _, cal, _, _ = metrics(eq, strat_ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_y : test_end]
        bull_t = get_bull_signal(qqq.loc[full_idx], f, s).astype(float)
        bull_test = bull_t.loc[test_idx]
        qpos = 1 - bull_test; tpos = bull_test
        qret = qqq.loc[test_idx].pct_change().fillna(0); tret = tqqq_full.loc[test_idx].pct_change().fillna(0)
        qpos_lag = qpos.shift(1).fillna(0); tpos_lag = tpos.shift(1).fillna(0)
        cost = (qpos_lag.diff().abs().fillna(0) + tpos_lag.diff().abs().fillna(0)) * 7.5/10000
        strat_ret = qpos_lag * qret + tpos_lag * tret - cost
        all_returns.append(strat_ret)
        start_idx += 252 * test_y
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret

eq, ret = wf_qqq_tqqq(qqq, tqqq_full, 5, 2)
c, m, sh, ca, fv, so = metrics(eq, ret)
print(f"{'★ WF QQQ/TQQQ 轮换 (前最佳, 含 QQQ)':<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# 同期 buy & hold
test_period = ret.index
tqqq_t = tqqq_full.loc[test_period]
qqq_t = qqq.loc[test_period]
for name, px in [("TQQQ B&H 同期", tqqq_t), ("QQQ B&H 同期", qqq_t)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

print()
print("=" * 100)
print("【冠军分析】纯 TQQQ 金字塔最佳 vs WF 轮换基准")
print("=" * 100)

best_calmar = max(wf_results.items(), key=lambda x: x[1][3])
best_cagr = max(wf_results.items(), key=lambda x: x[1][0])

print(f"\n最高 Calmar (纯 TQQQ): {best_calmar[0]}")
print(f"  CAGR {best_calmar[1][0]*100:.2f}%, MDD {best_calmar[1][1]*100:.2f}%, Calmar {best_calmar[1][3]:.3f}")
print(f"\n最高 CAGR (纯 TQQQ):   {best_cagr[0]}")
print(f"  CAGR {best_cagr[1][0]*100:.2f}%, MDD {best_cagr[1][1]*100:.2f}%, Calmar {best_cagr[1][3]:.3f}")

print()
print("=" * 100)
print("【过拟合验证】1999-2010 严格样本外测试")
print("=" * 100)

oos_idx = qqq.index <= "2010-02-10"
print(f"\n{'策略':<58} {'OOS CAGR':>10} {'OOS MDD':>10} {'OOS Calmar':>11}")
print("-" * 100)
for name, levels, base, anchor in wf_configs:
    bull_oos = get_bull_signal(qqq[oos_idx], 5, 200)
    tpos = build_pure_tqqq_pyramid(qqq[oos_idx], tqqq_full[oos_idx], bull_oos, 200, base, levels, anchor)
    eq, ret, _ = backtest_tqqq_cash(tpos, tqqq_full[oos_idx])
    c, m, sh, ca, fv, so = metrics(eq, ret)
    flag = "✅" if c > 0 else "❌"
    print(f"{name:<58} {c*100:>9.2f}% {m*100:>9.2f}% {ca:>11.3f} {flag}")

print()
print("=" * 100)
print("【Bootstrap】最优配置显著性")
print("=" * 100)

best_ret = best_calmar[1][7]
np.random.seed(42)
cagrs = []; sharpes = []
n_obs = len(best_ret); block = 20
for _ in range(1000):
    starts = np.random.randint(0, n_obs - block, n_obs // block)
    idx = np.concatenate([np.arange(st, st+block) for st in starts])
    idx = idx[idx < n_obs]
    s = best_ret.values[idx]
    eq = np.cumprod(1 + s)
    years = len(s) / 252
    if eq[-1] > 0: cagrs.append(eq[-1] ** (1/years) - 1)
    if s.std() > 0: sharpes.append(s.mean() * 252 / (s.std() * np.sqrt(252)))
cagrs, sharpes = np.array(cagrs), np.array(sharpes)
print(f"\n{best_calmar[0]}:")
print(f"  CAGR  中位数 {np.median(cagrs)*100:.2f}%, 95% CI [{np.percentile(cagrs, 2.5)*100:.2f}%, {np.percentile(cagrs, 97.5)*100:.2f}%]")
print(f"  Sharpe 中位数 {np.median(sharpes):.3f}, 95% CI [{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]")
print(f"  CAGR > 0 概率: {(cagrs > 0).mean()*100:.1f}%")
print(f"  CAGR > 20% 概率: {(cagrs > 0.20).mean()*100:.1f}%")
