"""
WF 轮换 + 现金储备金字塔抄底（按用户的正确理解）

设计:
- 牛市 (EMA 信号 up): 100% TQQQ
- 熊市 (EMA 信号 down): X% QQQ + (100-X)% 现金储备 (待命抄底)
- 跌破 MA200 越深, 用现金储备买越多 TQQQ 抄底:
  - 跌 MA200 -10%: 部署 25% 现金储备 → TQQQ
  - 跌 MA200 -20%: 部署 50% 现金储备 → TQQQ
  - 跌 MA200 -30%: 部署 75% 现金储备 → TQQQ
  - 跌 MA200 -40%: 部署 100% 现金储备 → TQQQ
- 信号反转回牛 (EMA 上穿): 卖 QQQ + 卖抄底 TQQQ → 全仓 TQQQ
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "1999-03-10"
END = "2026-04-18"
INIT_CASH = 10_000.0

print("[1/5] 数据准备 ...")
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
print(f"  QQQ {qqq.index[0].date()} ~ {qqq.index[-1].date()}, {len(qqq)} 天")
print(f"  TQQQ 起 ${tqqq_full.iloc[0]:.4f}, 终 ${tqqq_full.iloc[-1]:.2f}")

def backtest_3asset(qpos, tpos, cpos, qqq, tqqq, fee_bps=2.5, slip_bps=5.0):
    """
    qpos + tpos + cpos = 1.0 每日。cpos = 现金权重 (生息 0%)
    """
    qpos_lag = qpos.shift(1).fillna(0)
    tpos_lag = tpos.shift(1).fillna(0)
    qret = qqq.pct_change().fillna(0)
    tret = tqqq.pct_change().fillna(0)
    pos_change = qpos_lag.diff().abs().fillna(0) + tpos_lag.diff().abs().fillna(0)
    cost = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = qpos_lag * qret + tpos_lag * tret - cost
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

def wf_rotation_pyramid(qqq, tqqq_full, train_years=5, test_years=2,
                        bear_qqq_pct=0.60, bear_cash_pct=0.40,
                        pyramid_levels=None, ma_for_pyramid=200):
    """
    Walk-forward 轮换 + 现金储备金字塔
    - bear_qqq_pct: 熊市 QQQ 占比
    - bear_cash_pct: 熊市现金储备占比 (用于金字塔抄底)
    - pyramid_levels: [(drawdown_threshold, deploy_fraction)]
    """
    if pyramid_levels is None:
        pyramid_levels = [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)]

    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    chosen_params = []
    test_periods = []

    start_idx = 252 * train_years
    while start_idx + 252 * test_years <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_years, len(qqq))
        train_idx = qqq.index[train_end - 252*train_years : train_end]
        test_idx = qqq.index[train_end : test_end]

        # 在训练集选最优 (f, s)
        best_cal = -999
        best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                ema_f = qqq.loc[train_idx].ewm(span=f, adjust=False).mean()
                ema_s = qqq.loc[train_idx].ewm(span=s, adjust=False).mean()
                bull = (ema_f > ema_s)
                bull_arr = bull.astype(float); bull_arr.iloc[:s] = 0
                # 简化：用 100% TQQQ / 100% QQQ 选参数
                eq, ret, _ = backtest_3asset(
                    1 - bull_arr, bull_arr, pd.Series(0.0, index=train_idx),
                    qqq.loc[train_idx], tqqq_full.loc[train_idx]
                )
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)

        # 应用到 test 期 + 金字塔逻辑
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_years : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        ma_p = qqq.loc[full_idx].ewm(span=ma_for_pyramid, adjust=False).mean()

        bull = (ema_f > ema_s)
        bull.iloc[:s] = False

        # 计算每日仓位
        qpos = pd.Series(0.0, index=full_idx)
        tpos = pd.Series(0.0, index=full_idx)
        cpos = pd.Series(0.0, index=full_idx)

        for i in range(len(full_idx)):
            t = full_idx[i]
            if i < max(s, ma_for_pyramid):
                cpos.iloc[i] = 1.0
                continue

            if bull.iloc[i]:
                # 牛: 100% TQQQ
                qpos.iloc[i] = 0
                tpos.iloc[i] = 1.0
                cpos.iloc[i] = 0
            else:
                # 熊: bear_qqq_pct QQQ + bear_cash_pct 现金 (待金字塔抄底)
                # 计算价格相对 MA200 的偏离
                deviation = float(qqq.loc[t] / ma_p.loc[t] - 1)
                # 部署的现金比例 (按金字塔档)
                deployed_frac = 0.0
                for thresh, frac in pyramid_levels:
                    if deviation <= thresh:
                        deployed_frac = frac
                # 部署的现金 → TQQQ
                deployed_cash = bear_cash_pct * deployed_frac
                tpos.iloc[i] = deployed_cash
                cpos.iloc[i] = bear_cash_pct - deployed_cash
                qpos.iloc[i] = bear_qqq_pct

        # 在 test 期取出
        qpos_t = qpos.loc[test_idx]
        tpos_t = tpos.loc[test_idx]
        cpos_t = cpos.loc[test_idx]

        eq, ret, _ = backtest_3asset(
            qpos_t, tpos_t, cpos_t,
            qqq.loc[test_idx], tqqq_full.loc[test_idx]
        )
        all_returns.append(ret)
        chosen_params.append(best_p)
        test_periods.append((test_idx[0], test_idx[-1]))
        start_idx += 252 * test_years

    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret, chosen_params, test_periods


print()
print("=" * 100)
print("【实验】WF 轮换 + 现金储备金字塔 — 多种配置对比")
print("=" * 100)

# 对照: 之前最佳 WF 轮换
def wf_pure_rotation(qqq, tqqq_full, train_years=5, test_years=2):
    fast_grid = [3, 5, 8, 10, 13]
    slow_grid = [50, 100, 150, 200, 250]
    all_returns = []
    start_idx = 252 * train_years
    while start_idx + 252 * test_years <= len(qqq):
        train_end = start_idx
        test_end = min(start_idx + 252 * test_years, len(qqq))
        train_idx = qqq.index[train_end - 252*train_years : train_end]
        test_idx = qqq.index[train_end : test_end]
        best_cal = -999
        best_p = (5, 200)
        for f in fast_grid:
            for s in slow_grid:
                if f >= s: continue
                ema_f = qqq.loc[train_idx].ewm(span=f, adjust=False).mean()
                ema_s = qqq.loc[train_idx].ewm(span=s, adjust=False).mean()
                bull = (ema_f > ema_s).astype(float); bull.iloc[:s] = 0
                eq, ret, _ = backtest_3asset(1 - bull, bull, pd.Series(0.0, index=train_idx), qqq.loc[train_idx], tqqq_full.loc[train_idx])
                _, _, _, cal, _, _ = metrics(eq, ret)
                if cal > best_cal:
                    best_cal = cal; best_p = (f, s)
        f, s = best_p
        full_idx = qqq.index[train_end - 252*train_years : test_end]
        ema_f = qqq.loc[full_idx].ewm(span=f, adjust=False).mean()
        ema_s = qqq.loc[full_idx].ewm(span=s, adjust=False).mean()
        bull = (ema_f > ema_s).astype(float); bull.iloc[:s] = 0
        bull_t = bull.loc[test_idx]
        eq, ret, _ = backtest_3asset(1 - bull_t, bull_t, pd.Series(0.0, index=test_idx), qqq.loc[test_idx], tqqq_full.loc[test_idx])
        all_returns.append(ret)
        start_idx += 252 * test_years
    full_ret = pd.concat(all_returns)
    full_eq = (1 + full_ret).cumprod() * INIT_CASH
    return full_eq, full_ret

print("\n基准（之前最佳 WF QQQ/TQQQ 轮换）...")
base_eq, base_ret = wf_pure_rotation(qqq, tqqq_full)
c, m, sh, ca, fv, so = metrics(base_eq, base_ret)
print(f"  CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Sharpe {sh:.3f}, Sortino {so:.3f}, Calmar {ca:.3f}, 终值 ${fv:,.0f}")

# 多种金字塔配置
configs = {
    "P1 60Q/40C, 抄底 -10/-20/-30/-40 → 25/50/75/100%": (
        0.60, 0.40, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P2 70Q/30C, 抄底 -10/-20/-30/-40 → 25/50/75/100%": (
        0.70, 0.30, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P3 50Q/50C, 抄底 -10/-20/-30/-40 → 25/50/75/100%": (
        0.50, 0.50, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P4 60Q/40C, 激进 -5/-15/-25/-35 → 25/50/75/100%": (
        0.60, 0.40, [(-0.05, 0.25), (-0.15, 0.50), (-0.25, 0.75), (-0.35, 1.00)], 200),
    "P5 60Q/40C, 保守 -15/-25/-35/-45 → 25/50/75/100%": (
        0.60, 0.40, [(-0.15, 0.25), (-0.25, 0.50), (-0.35, 0.75), (-0.45, 1.00)], 200),
    "P6 60Q/40C, 早部署 -5/-10/-20/-30 → 25/50/75/100%": (
        0.60, 0.40, [(-0.05, 0.25), (-0.10, 0.50), (-0.20, 0.75), (-0.30, 1.00)], 200),
    "P7 80Q/20C, 抄底 -10/-20/-30/-40 → 25/50/75/100%": (
        0.80, 0.20, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P8 0Q/100C, 全现金待命 (极端版)": (
        0.0, 1.0, [(-0.10, 0.25), (-0.20, 0.50), (-0.30, 0.75), (-0.40, 1.00)], 200),
    "P9 60Q/40C, 简化 -20/-40 → 50/100%": (
        0.60, 0.40, [(-0.20, 0.50), (-0.40, 1.00)], 200),
    "P10 60Q/40C, 5档 -10/-20/-30/-40/-50 → 20/40/60/80/100%": (
        0.60, 0.40, [(-0.10, 0.20), (-0.20, 0.40), (-0.30, 0.60), (-0.40, 0.80), (-0.50, 1.00)], 200),
}

print(f"\n{'金字塔配置':<58} {'CAGR':>7} {'MDD':>9} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'终值':>14}")
print("-" * 130)

results = {}
for name, (qp, cp, levels, ma) in configs.items():
    eq, ret, params, periods = wf_rotation_pyramid(
        qqq, tqqq_full, 5, 2,
        bear_qqq_pct=qp, bear_cash_pct=cp,
        pyramid_levels=levels, ma_for_pyramid=ma
    )
    c, m, sh, ca, fv, so = metrics(eq, ret)
    results[name] = (c, m, sh, ca, fv, so, eq, ret)
    print(f"{name:<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# 添加基准
print("-" * 130)
c, m, sh, ca, fv, so = metrics(base_eq, base_ret)
print(f"{'⭐基准: WF QQQ/TQQQ 轮换 (无金字塔)':<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# 同期 buy & hold
test_period = base_ret.index
qqq_t = qqq.loc[test_period]
tqqq_t = tqqq_full.loc[test_period]
for name, px in [("TQQQ B&H 同期", tqqq_t), ("QQQ B&H 同期", qqq_t)]:
    eq = (px / px.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    c, m, sh, ca, fv, so = metrics(eq, ret)
    print(f"{name:<58} {c*100:>6.2f}% {m*100:>8.2f}% {sh:>7.3f} {so:>8.3f} {ca:>7.3f} {fv:>14,.0f}")

# 找出最佳
print()
print("=" * 100)
print("【冠军分析】")
print("=" * 100)
best_by_calmar = max(results.items(), key=lambda x: x[1][3])
best_by_cagr = max(results.items(), key=lambda x: x[1][0])
best_by_sharpe = max(results.items(), key=lambda x: x[1][2])

print(f"\n最高 Calmar: {best_by_calmar[0]}")
print(f"  CAGR {best_by_calmar[1][0]*100:.2f}%, MDD {best_by_calmar[1][1]*100:.2f}%, Calmar {best_by_calmar[1][3]:.3f}")

print(f"\n最高 CAGR:   {best_by_cagr[0]}")
print(f"  CAGR {best_by_cagr[1][0]*100:.2f}%, MDD {best_by_cagr[1][1]*100:.2f}%, Calmar {best_by_cagr[1][3]:.3f}")

print(f"\n最高 Sharpe: {best_by_sharpe[0]}")
print(f"  CAGR {best_by_sharpe[1][0]*100:.2f}%, MDD {best_by_sharpe[1][1]*100:.2f}%, Sharpe {best_by_sharpe[1][2]:.3f}")

# 关键对比
print()
print("=" * 100)
print("【vs 基准对比】最佳金字塔 vs WF 纯轮换")
print("=" * 100)
c0, m0, sh0, ca0, fv0, so0 = metrics(base_eq, base_ret)
cb, mb, shb, cab, fvb, sob = best_by_calmar[1][:6]
print(f"\n基准 WF 轮换:                       CAGR {c0*100:.2f}%, MDD {m0*100:.2f}%, Sharpe {sh0:.3f}, Calmar {ca0:.3f}, 终值 ${fv0:,.0f}")
print(f"最佳金字塔 ({best_by_calmar[0][:30]}...):")
print(f"                                    CAGR {cb*100:.2f}%, MDD {mb*100:.2f}%, Sharpe {shb:.3f}, Calmar {cab:.3f}, 终值 ${fvb:,.0f}")
print(f"\n变化:  CAGR {(cb-c0)*100:+.2f}pp,  MDD {(mb-m0)*100:+.2f}pp,  Calmar {cab-ca0:+.3f},  终值 {(fvb/fv0-1)*100:+.1f}%")

# Bootstrap 显著性
print()
print("=" * 100)
print("【Bootstrap 1000 次】最佳金字塔显著性")
print("=" * 100)

best_ret = best_by_calmar[1][7]
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
print(f"\n{best_by_calmar[0]}:")
print(f"  CAGR  中位数 {np.median(cagrs)*100:.2f}%, 95% CI [{np.percentile(cagrs, 2.5)*100:.2f}%, {np.percentile(cagrs, 97.5)*100:.2f}%]")
print(f"  Sharpe 中位数 {np.median(sharpes):.3f}, 95% CI [{np.percentile(sharpes, 2.5):.3f}, {np.percentile(sharpes, 97.5):.3f}]")
print(f"  CAGR > 0 概率: {(cagrs > 0).mean()*100:.1f}%")
print(f"  CAGR > 20% 概率: {(cagrs > 0.20).mean()*100:.1f}%")
print(f"  CAGR > 25% 概率: {(cagrs > 0.25).mean()*100:.1f}%")
