"""
合成 TQQQ 1999-2026 数据 + 全样本回测 + 参数稳定性分析

合成公式: TQQQ_ret = 3 × QQQ_ret - daily_cost
- expense ratio: 0.84%/year
- financing: ~2 × short-term rate
- 验证: 用合成 2010-2026 与真实 TQQQ 比对，看是否吻合
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "1999-03-10"  # QQQ 上市日
END = "2026-01-17"
INIT_CASH = 10_000.0

print("[1/5] 下载 QQQ 全历史 + 真实 TQQQ ...")
qqq_full = yf.download("QQQ", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
tqqq_real = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
# 拉短期利率代理 (DTB3 = 3-month T-bill, 用 SHV ETF 当代理；这里直接用近似常数)
# 简化: 2000-2008 平均 ~3%, 2009-2015 ~0.1%, 2016-2019 ~1.5%, 2020-2021 ~0.1%, 2022-2026 ~4.5%
def daily_financing(date):
    y = date.year
    if y <= 2007: rate = 0.04
    elif y <= 2008: rate = 0.02
    elif y <= 2015: rate = 0.001
    elif y <= 2019: rate = 0.015
    elif y <= 2021: rate = 0.001
    else: rate = 0.045
    # 2x 借入 + 40bp spread
    return (rate + 0.004) * 2 / 252

print("[2/5] 合成 TQQQ ...")
qqq_ret = qqq_full.pct_change().fillna(0)
expense_daily = 0.0084 / 252
financing_daily = pd.Series([daily_financing(d) for d in qqq_full.index], index=qqq_full.index)
slip_daily = 0.0005  # 5 bp/day rebalancing slippage

tqqq_synth_ret = 3 * qqq_ret - expense_daily - financing_daily - slip_daily / 252 * 252  # = - slip_daily
# 第一天初始价 = $1
tqqq_synth = (1 + tqqq_synth_ret).cumprod()

print("[3/5] 验证合成 vs 真实 TQQQ (2010-2026 重叠期) ...")
overlap_start = tqqq_real.index[0]
synth_overlap = tqqq_synth.loc[overlap_start:]
real_overlap = tqqq_real.loc[overlap_start:]
# 对齐起点，归一化
synth_norm = synth_overlap / synth_overlap.iloc[0]
real_norm = real_overlap / real_overlap.iloc[0]

# 对比终值
final_synth = float(synth_norm.iloc[-1])
final_real = float(real_norm.iloc[-1])
print(f"  合成 TQQQ {overlap_start.date()}→今: {final_synth:.1f}x")
print(f"  真实 TQQQ {overlap_start.date()}→今: {final_real:.1f}x")
print(f"  误差: {abs(final_synth - final_real)/final_real*100:.1f}%")

# 用合成 + 真实拼起来：1999-2010 用合成，2010+ 用真实（按真实起点对齐）
scale = float(real_overlap.iloc[0]) / float(tqqq_synth.loc[overlap_start])
tqqq_pre2010 = tqqq_synth.loc[:overlap_start] * scale
tqqq_combined = pd.concat([tqqq_pre2010.iloc[:-1], tqqq_real])
print(f"  拼接后 TQQQ 序列: {tqqq_combined.index[0].date()} ~ {tqqq_combined.index[-1].date()}, {len(tqqq_combined)} 天")

print()
print("[4/5] 回测：EMA5/200 TQQQ-sig → close[T+1] 在 1999-2026 全样本")
print("=" * 100)

def backtest_clean(close_sig, close_exec, fast, slow, fee_bps=2.5, slip_bps=5.0):
    ema_f = close_sig.ewm(span=fast, adjust=False).mean()
    ema_s = close_sig.ewm(span=slow, adjust=False).mean()
    sig = (ema_f > ema_s).astype(int)
    sig.iloc[:slow] = 0
    pos = sig.shift(1).fillna(0)  # 次日持仓
    daily_ret = close_exec.pct_change().fillna(0)
    # 交易成本: 每次仓位变化扣一次 (fee + slip) bps
    pos_change = pos.diff().abs().fillna(0)
    cost_daily = pos_change * (fee_bps + slip_bps) / 10000
    strat_ret = pos * daily_ret - cost_daily
    eq = (1 + strat_ret).cumprod() * INIT_CASH
    n_trades = int((pos.diff().abs() > 0).sum())
    return eq, strat_ret, n_trades

def metrics(eq, ret, label):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sh = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {"label": label, "cagr": cagr, "mdd": mdd, "sharpe": sh, "calmar": cal, "fv": eq.iloc[-1]}

# 全样本 1999-2026 用合成+真实拼接
periods = {
    "全样本 1999-2026 (含 dot-com + 2008 + COVID + 2022)": tqqq_combined,
    "样本外 1999-2010 (dot-com + 2008)": tqqq_combined.loc[:"2010-02-10"],
    "样本内 2010-2026 (真实 TQQQ)": tqqq_real,
}

# 对比 buy & hold
def bh_metrics(price, label):
    eq = (price / price.iloc[0]) * INIT_CASH
    ret = eq.pct_change().fillna(0)
    return metrics(eq, ret, label)

results_by_period = {}
for period_name, px in periods.items():
    print(f"\n--- {period_name} ---")
    print(f"{'策略':<40} {'CAGR':>9} {'MDD':>9} {'夏普':>8} {'卡玛':>8} {'终值':>14} {'交易':>5}")
    print("-" * 100)
    eq, ret, n = backtest_clean(px, px, 5, 200)
    m = metrics(eq, ret, "EMA5/200 self-sig"); m['n'] = n
    print(f"{m['label']:<40} {m['cagr']*100:>8.2f}% {m['mdd']*100:>8.2f}% {m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['fv']:>14,.0f} {n:>5}")
    bh = bh_metrics(px, "TQQQ Buy & Hold")
    print(f"{bh['label']:<40} {bh['cagr']*100:>8.2f}% {bh['mdd']*100:>8.2f}% {bh['sharpe']:>8.3f} {bh['calmar']:>8.3f} {bh['fv']:>14,.0f} {'-':>5}")
    results_by_period[period_name] = (m, bh)

print()
print("[5/5] 参数稳定性扫描：fast × slow 网格（避免过拟合的关键检验）")
print("=" * 100)
print("\n用 1999-2026 全样本（关键：在样本外稳定的参数才是好参数）")

fast_grid = [3, 5, 8, 10, 13, 20]
slow_grid = [50, 100, 150, 200, 250, 300]

print(f"\n{'CAGR%':<8}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
for f in fast_grid:
    print(f"f={f:<5}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        try:
            eq, ret, n = backtest_clean(tqqq_combined, tqqq_combined, f, s)
            m = metrics(eq, ret, "")
            print(f"{m['cagr']*100:>8.1f}%", end="")
        except Exception as e:
            print(f"{'err':>9}", end="")
    print()

print(f"\n{'MDD%':<8}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
for f in fast_grid:
    print(f"f={f:<5}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        try:
            eq, ret, n = backtest_clean(tqqq_combined, tqqq_combined, f, s)
            m = metrics(eq, ret, "")
            print(f"{m['mdd']*100:>8.1f}%", end="")
        except: pass
    print()

print(f"\n{'Calmar':<8}", end="")
for s in slow_grid: print(f"{s:>9}", end="")
print()
for f in fast_grid:
    print(f"f={f:<5}", end="")
    for s in slow_grid:
        if f >= s:
            print(f"{'-':>9}", end="")
            continue
        try:
            eq, ret, n = backtest_clean(tqqq_combined, tqqq_combined, f, s)
            m = metrics(eq, ret, "")
            print(f"{m['calmar']:>9.3f}", end="")
        except: pass
    print()

print()
print("=" * 100)
print("【关键解读】")
print("=" * 100)
print("- 看 CAGR/Calmar 表格：理想参数应该在'高原'(plateau)而不是孤立尖峰")
print("- 如果某组合 CAGR 特别高但邻居都很差，那就是过拟合")
print("- 真正稳健的参数：邻居们 (f±2, s±50) 也都还不错")
