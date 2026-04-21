"""
合成 TQQQ v2 - 修正成本模型 + 严格防过拟合分析
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "1999-03-10"
END = "2026-01-17"
INIT_CASH = 10_000.0

print("[1/6] 下载 QQQ 全历史 + 真实 TQQQ ...")
qqq_full = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()

# 修正合成模型
def daily_financing(date):
    """美联储基金利率年化 → 日"""
    y = date.year
    if y <= 2007: rate = 0.045
    elif y <= 2008: rate = 0.025
    elif y <= 2015: rate = 0.0015
    elif y <= 2019: rate = 0.015
    elif y <= 2021: rate = 0.001
    else: rate = 0.045
    # 借入 2x 本金来达到 3x 杠杆，financing = rate × 2
    return (rate + 0.004) * 2 / 252  # +40bp spread

print("[2/6] 合成 TQQQ (修正成本)")
qqq_ret = qqq_full.pct_change().fillna(0)
expense_daily = 0.0084 / 252  # 0.84%/年
financing_daily = pd.Series([daily_financing(d) for d in qqq_full.index], index=qqq_full.index)
slip_annual = 0.003  # 0.3%/年的再平衡滑点（保守估计）
slip_daily = slip_annual / 252

tqqq_synth_ret = 3 * qqq_ret - expense_daily - financing_daily - slip_daily
tqqq_synth = (1 + tqqq_synth_ret).cumprod()

# 校验合成 vs 真实
overlap_start = tqqq_real.index[0]
synth_overlap = tqqq_synth.loc[overlap_start:]
real_overlap = tqqq_real.loc[overlap_start:]
synth_norm = synth_overlap / synth_overlap.iloc[0]
real_norm = real_overlap / real_overlap.iloc[0]
final_synth = float(synth_norm.iloc[-1])
final_real = float(real_norm.iloc[-1])
err = abs(final_synth - final_real) / final_real * 100
print(f"  合成 TQQQ 16年: {final_synth:.1f}x")
print(f"  真实 TQQQ 16年: {final_real:.1f}x")
print(f"  误差: {err:.1f}%")

# 仍有误差就用乘法校准: 让 16 年终值匹配
calibration = final_real / final_synth
implied_extra_drag = (calibration ** (-1/16) - 1) * 100
print(f"  隐含额外年化漂移: {implied_extra_drag:+.2f}%/年")
# 应用校准
tqqq_synth_calibrated = tqqq_synth * (calibration ** (np.arange(len(tqqq_synth)) / len(tqqq_synth)))
synth_cal_final = float(tqqq_synth_calibrated.loc[overlap_start:].iloc[-1] / tqqq_synth_calibrated.loc[overlap_start])
print(f"  校准后合成 16年: {synth_cal_final:.1f}x （目标 {final_real:.1f}x）")

# 拼接：1999-2010 用校准合成，2010+ 用真实
scale = float(real_overlap.iloc[0]) / float(tqqq_synth_calibrated.loc[overlap_start])
tqqq_pre2010 = tqqq_synth_calibrated.loc[:overlap_start] * scale
tqqq_combined = pd.concat([tqqq_pre2010.iloc[:-1], tqqq_real])
print(f"  拼接后 TQQQ: {tqqq_combined.index[0].date()} ~ {tqqq_combined.index[-1].date()}, {len(tqqq_combined)} 天")
print()

def backtest_clean(close_sig, close_exec, fast, slow, fee_bps=2.5, slip_bps=5.0):
    ema_f = close_sig.ewm(span=fast, adjust=False).mean()
    ema_s = close_sig.ewm(span=slow, adjust=False).mean()
    sig = (ema_f > ema_s).astype(int)
    sig.iloc[:slow] = 0
    pos = sig.shift(1).fillna(0)
    daily_ret = close_exec.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    n_trades = int((pos.diff().abs() > 0).sum())
    return eq, strat_ret, n_trades

def metrics(eq, ret):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return cagr, mdd, sh, cal, eq.iloc[-1]

print("[3/6] 三段式样本测试：样本内/样本外/全样本")
print("=" * 100)

periods = {
    "样本外: 1999-2010 (dot-com -82%, 2008 -56%)": tqqq_combined.loc[:"2010-02-10"],
    "样本内: 2010-2026 (大牛市)": tqqq_real,
    "全样本: 1999-2026 (跨多周期)": tqqq_combined,
}

print(f"\n{'时段':<48} {'策略 CAGR':>10} {'策略 MDD':>10} {'卡玛':>7} {'B&H CAGR':>10} {'B&H MDD':>10}")
print("-" * 110)
for period_name, px in periods.items():
    eq, ret, n = backtest_clean(px, px, 5, 200)
    s_cagr, s_mdd, _, s_cal, _ = metrics(eq, ret)
    bh_eq = (px / px.iloc[0]) * INIT_CASH
    bh_ret = bh_eq.pct_change().fillna(0)
    b_cagr, b_mdd, _, _, _ = metrics(bh_eq, bh_ret)
    print(f"{period_name:<48} {s_cagr*100:>9.2f}% {s_mdd*100:>9.2f}% {s_cal:>7.3f} {b_cagr*100:>9.2f}% {b_mdd*100:>9.2f}%")

print()
print("[4/6] 参数稳定性热力图（防过拟合关键）")
print("=" * 100)
print("规则：理想参数应在'高原 plateau'，邻居 (f±2, s±50) 也都接近")
print("如果某点 Calmar 远高于邻居 → 过拟合警报")
print()

fast_grid = [3, 5, 8, 10, 13, 20, 30]
slow_grid = [50, 100, 150, 200, 250, 300]

# 用全样本（含两次崩盘）评估稳定性
test_px = tqqq_combined

print("--- 全样本 1999-2026 Calmar ratio ---")
print(f"{'fast\\slow':<10}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
calmar_grid = {}
for f in fast_grid:
    print(f"{f:<10}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        eq, ret, n = backtest_clean(test_px, test_px, f, s)
        _, _, _, cal, _ = metrics(eq, ret)
        calmar_grid[(f, s)] = cal
        print(f"{cal:>9.3f}", end="")
    print()

print()
print("--- 仅样本外 1999-2010 Calmar (真正的过拟合检测) ---")
test_px2 = tqqq_combined.loc[:"2010-02-10"]
print(f"{'fast\\slow':<10}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
calmar_oos = {}
for f in fast_grid:
    print(f"{f:<10}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        eq, ret, n = backtest_clean(test_px2, test_px2, f, s)
        _, _, _, cal, _ = metrics(eq, ret)
        calmar_oos[(f, s)] = cal
        print(f"{cal:>9.3f}", end="")
    print()

print()
print("[5/6] 跨资产稳健性（同样参数在 SPY/UPRO 上能赢吗？）")
print("=" * 100)
spy = yf.download("SPY", start="1999-03-10", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
upro = yf.download("UPRO", start="2009-06-25", end=END, auto_adjust=True, progress=False)["Close"].squeeze()

for name, px in [("SPY (1× S&P)", spy), ("UPRO (3× S&P)", upro)]:
    eq, ret, n = backtest_clean(px, px, 5, 200)
    s_cagr, s_mdd, s_sh, s_cal, fv = metrics(eq, ret)
    bh_eq = (px / px.iloc[0]) * INIT_CASH
    bh_ret = bh_eq.pct_change().fillna(0)
    b_cagr, b_mdd, b_sh, b_cal, _ = metrics(bh_eq, bh_ret)
    print(f"\n{name} ({px.index[0].date()} ~ {px.index[-1].date()})")
    print(f"  EMA5/200:    CAGR {s_cagr*100:>6.2f}%, MDD {s_mdd*100:>6.2f}%, Calmar {s_cal:.3f}, 交易 {n}")
    print(f"  Buy & Hold:  CAGR {b_cagr*100:>6.2f}%, MDD {b_mdd*100:>6.2f}%, Calmar {b_cal:.3f}")

print()
print("[6/6] 6 种实用防过拟合优化方法（给出可执行版本）")
print("=" * 100)

# 优化 1: VIX 滤波（用 ^VIX）
print("\n【优化 1】VIX 滤波 — VIX > 30 时强制空仓")
vix = yf.download("^VIX", start="1999-03-10", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
vix = vix.reindex(tqqq_combined.index).ffill()

def backtest_with_vix(px, fast, slow, vix_thresh=30):
    ema_f = px.ewm(span=fast, adjust=False).mean()
    ema_s = px.ewm(span=slow, adjust=False).mean()
    sig = ((ema_f > ema_s) & (vix < vix_thresh)).astype(int)
    sig.iloc[:slow] = 0
    pos = sig.shift(1).fillna(0)
    daily_ret = px.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * 7.5 / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret, int((pos.diff().abs() > 0).sum())

eq, ret, n = backtest_with_vix(tqqq_combined, 5, 200, 30)
c, m, sh, cal, fv = metrics(eq, ret)
print(f"  全样本 1999-2026: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {cal:.3f}, 交易 {n}")

# 优化 2: 仓位 sizing 而不是 100%/0%
print("\n【优化 2】部分仓位（70% 持仓而不是 100%）减少 TQQQ 杠杆暴露")
def backtest_partial(px, fast, slow, pos_size=0.7):
    ema_f = px.ewm(span=fast, adjust=False).mean()
    ema_s = px.ewm(span=slow, adjust=False).mean()
    sig = (ema_f > ema_s).astype(float) * pos_size
    sig.iloc[:slow] = 0
    pos = sig.shift(1).fillna(0)
    daily_ret = px.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * 7.5 / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret

eq, ret = backtest_partial(tqqq_combined, 5, 200, 0.7)
c, m, sh, cal, fv = metrics(eq, ret)
print(f"  70% 仓位: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {cal:.3f}")

# 优化 3: 集成多参数 (ensemble)
print("\n【优化 3】多 EMA 集成（5/100 + 5/200 + 10/200 各 1/3 仓位）")
def backtest_ensemble(px, params_list):
    sigs = []
    for f, s in params_list:
        ema_f = px.ewm(span=f, adjust=False).mean()
        ema_s = px.ewm(span=s, adjust=False).mean()
        sig = (ema_f > ema_s).astype(float)
        sig.iloc[:s] = 0
        sigs.append(sig)
    avg_sig = sum(sigs) / len(sigs)
    pos = avg_sig.shift(1).fillna(0)
    daily_ret = px.pct_change().fillna(0)
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * 7.5 / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    return eq, strat_ret

eq, ret = backtest_ensemble(tqqq_combined, [(5, 100), (5, 200), (10, 200)])
c, m, sh, cal, fv = metrics(eq, ret)
print(f"  3 信号集成: CAGR {c*100:.2f}%, MDD {m*100:.2f}%, Calmar {cal:.3f}")
