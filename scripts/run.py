"""
EMA 5/200 crossover strategy backtest - 4 version comparison
- Signal source: QQQ vs TQQQ
- Trading instrument: TQQQ (unified)
- Cross up: full buy, cross down: full sell (hold nothing when flat)
- Compare vs buy & hold TQQQ / QQQ
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "2010-02-11"
END = "2026-01-17"
INIT_CASH = 10_000.0

print(f"[1/4] Downloading data {START} -> {END} ...")
data = yf.download(["QQQ", "TQQQ"], start=START, end=END, auto_adjust=True, progress=False)
close = data["Close"].dropna()
print(f"  Data range: {close.index[0].date()} ~ {close.index[-1].date()}, {len(close)} trading days")

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def backtest(signal_series, trade_price_series, init_cash=INIT_CASH, label=""):
    """Signal: EMA5 crosses above EMA200 -> full buy, crosses below -> full sell. Next-day open fill."""
    ema5 = ema(signal_series, 5)
    ema200 = ema(signal_series, 200)
    # Signal: 1 = long, 0 = cash
    sig = (ema5 > ema200).astype(int)
    # Cross up / cross down events
    cross_up = (sig == 1) & (sig.shift(1) == 0)
    cross_dn = (sig == 0) & (sig.shift(1) == 1)
    # T+1 execution: today's signal, next day fill at trade_price
    trade_signal = sig.shift(1).fillna(0)  # today's holding state
    # 200 EMA needs warm-up, skip first 200 days
    valid = trade_signal.index >= trade_signal.index[200]
    trade_signal = trade_signal[valid]
    px = trade_price_series.loc[trade_signal.index]
    daily_ret = px.pct_change().fillna(0)
    strat_ret = daily_ret * trade_signal
    equity = (1 + strat_ret).cumprod() * init_cash
    # Trade count
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
        "Total Return": f"{total_ret*100:.2f}%",
        "CAGR": f"{cagr*100:.2f}%",
        "Max Drawdown": f"{max_dd*100:.2f}%",
        "Volatility": f"{vol*100:.2f}%",
        "Sharpe": f"{sharpe:.3f}",
        "Sortino": f"{sortino:.3f}",
        "Calmar": f"{calmar:.3f}",
        "Final Value": f"${equity.iloc[-1]:,.0f}",
    }

print("[2/4] Running backtest ...")
qqq = close["QQQ"]
tqqq = close["TQQQ"]

# 4 strategies
results = []

# 1. EMA(QQQ) -> trade TQQQ
eq1, ret1, b1, s1 = backtest(qqq, tqqq, label="A")
results.append((metrics(eq1, ret1, "A. EMA5/200 QQQ signal -> trade TQQQ"), b1, s1))

# 2. EMA(TQQQ) -> trade TQQQ
eq2, ret2, b2, s2 = backtest(tqqq, tqqq, label="B")
results.append((metrics(eq2, ret2, "B. EMA5/200 TQQQ signal -> trade TQQQ"), b2, s2))

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

print("[3/4] Results summary:")
print()
print(f"{'Strategy':<42} {'CAGR':>10} {'Max DD':>12} {'Sharpe':>8} {'Calmar':>8} {'Final Value':>14} {'Trades':>10}")
print("-" * 110)
for m, b, s in results:
    n_trades = b + s
    print(f"{m['label']:<42} {m['CAGR']:>10} {m['Max Drawdown']:>12} {m['Sharpe']:>8} {m['Calmar']:>8} {m['Final Value']:>14} {n_trades:>10}")

print()
print("[4/4] vs moomoo screenshot benchmark:")
print(f"{'moomoo (new strategy 1)':<42} {'47.08%':>10} {'-72.80%':>12} {'1.083':>8} {'0.647':>8} {'$425,895':>14} {'53':>10}")

print()
print("Detailed metrics:")
for m, b, s in results:
    print(f"\n--- {m['label']} ---")
    for k, v in m.items():
        if k != "label":
            print(f"  {k}: {v}")
    print(f"  Buys: {b}, Sells: {s}, Total trades: {b+s}")
