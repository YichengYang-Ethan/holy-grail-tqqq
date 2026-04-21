"""
EMA 5/200 交叉策略回测 - 4 个版本对比
- 信号源: QQQ vs TQQQ
- 交易标的: TQQQ (统一)
- 上穿全仓买入, 下穿全仓卖出 (空仓时不持有)
- 对比 buy & hold TQQQ / QQQ
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "2010-02-11"
END = "2026-01-17"
INIT_CASH = 10_000.0

print(f"[1/4] 下载数据 {START} -> {END} ...")
data = yf.download(["QQQ", "TQQQ"], start=START, end=END, auto_adjust=True, progress=False)
close = data["Close"].dropna()
print(f"  数据范围: {close.index[0].date()} ~ {close.index[-1].date()}, {len(close)} 个交易日")

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def backtest(signal_series, trade_price_series, init_cash=INIT_CASH, label=""):
    """信号: EMA5 上穿 EMA200 全仓买, 下穿全仓卖。次日开盘 fill。"""
    ema5 = ema(signal_series, 5)
    ema200 = ema(signal_series, 200)
    # 信号: 1 = long, 0 = cash
    sig = (ema5 > ema200).astype(int)
    # 上穿/下穿事件
    cross_up = (sig == 1) & (sig.shift(1) == 0)
    cross_dn = (sig == 0) & (sig.shift(1) == 1)
    # T+1 执行: 今天信号, 明天用 trade_price 成交
    trade_signal = sig.shift(1).fillna(0)  # 今天持有状态
    # 200 EMA 需要预热, 跳过前 200 天
    valid = trade_signal.index >= trade_signal.index[200]
    trade_signal = trade_signal[valid]
    px = trade_price_series.loc[trade_signal.index]
    daily_ret = px.pct_change().fillna(0)
    strat_ret = daily_ret * trade_signal
    equity = (1 + strat_ret).cumprod() * init_cash
    # 交易次数
    n_buy = int(((trade_signal == 1) & (trade_signal.shift(1) == 0)).sum())
    n_sell = int(((trade_signal == 0) & (trade_signal.shift(1) == 1)).sum())
    return equity, strat_ret, n_buy, n_sell

def metrics(equity, daily_ret, label):
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max - 1)
    max_dd = drawdown.min()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = (daily_ret.mean() * 252) / downside if downside > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0
    return {
        "label": label,
        "总收益": f"{total_ret*100:.2f}%",
        "年化": f"{cagr*100:.2f}%",
        "最大回撤": f"{max_dd*100:.2f}%",
        "波动率": f"{vol*100:.2f}%",
        "夏普": f"{sharpe:.3f}",
        "索提诺": f"{sortino:.3f}",
        "卡玛": f"{calmar:.3f}",
        "终值": f"${equity.iloc[-1]:,.0f}",
    }

print("[2/4] 跑回测 ...")
qqq = close["QQQ"]
tqqq = close["TQQQ"]

# 4 个策略
results = []

# 1. EMA(QQQ) → trade TQQQ
eq1, ret1, b1, s1 = backtest(qqq, tqqq, label="A")
results.append((metrics(eq1, ret1, "A. EMA5/200 用 QQQ 信号 → 交易 TQQQ"), b1, s1))

# 2. EMA(TQQQ) → trade TQQQ
eq2, ret2, b2, s2 = backtest(tqqq, tqqq, label="B")
results.append((metrics(eq2, ret2, "B. EMA5/200 用 TQQQ 信号 → 交易 TQQQ"), b2, s2))

# 3. TQQQ buy & hold
tqqq_aligned = tqqq.loc[eq1.index]
bh_tqqq = (tqqq_aligned / tqqq_aligned.iloc[0]) * INIT_CASH
ret_bh_t = tqqq_aligned.pct_change().fillna(0)
results.append((metrics(bh_tqqq, ret_bh_t, "C. TQQQ Buy & Hold"), 1, 0))

# 4. QQQ buy & hold
qqq_aligned = qqq.loc[eq1.index]
bh_qqq = (qqq_aligned / qqq_aligned.iloc[0]) * INIT_CASH
ret_bh_q = qqq_aligned.pct_change().fillna(0)
results.append((metrics(bh_qqq, ret_bh_q, "D. QQQ Buy & Hold"), 1, 0))

print("[3/4] 结果汇总:")
print()
print(f"{'策略':<42} {'年化':>10} {'最大回撤':>12} {'夏普':>8} {'卡玛':>8} {'终值':>14} {'交易次数':>10}")
print("-" * 110)
for m, b, s in results:
    n_trades = b + s
    print(f"{m['label']:<42} {m['年化']:>10} {m['最大回撤']:>12} {m['夏普']:>8} {m['卡玛']:>8} {m['终值']:>14} {n_trades:>10}")

print()
print("[4/4] vs moomoo 截图基准:")
print(f"{'moomoo (新建策略1)':<42} {'47.08%':>10} {'-72.80%':>12} {'1.083':>8} {'0.647':>8} {'$425,895':>14} {'53':>10}")

print()
print("详细指标:")
for m, b, s in results:
    print(f"\n--- {m['label']} ---")
    for k, v in m.items():
        if k != "label":
            print(f"  {k}: {v}")
    print(f"  买入次数: {b}, 卖出次数: {s}, 总交易: {b+s}")
