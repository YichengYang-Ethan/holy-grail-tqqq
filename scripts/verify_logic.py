"""
Strict code logic verification:
1. Execute at OPEN price (signal close[T] -> buy/sell open[T+1]) - no look-ahead
2. Execute at CLOSE price (signal close[T] -> buy/sell close[T+1]) - no look-ahead
3. Include commission + slippage
4. Verify EMA computation and cross detection
"""
import yfinance as yf
import pandas as pd
import numpy as np

START = "2010-02-11"
END = "2026-01-17"
INIT_CASH = 10_000.0
FAST, SLOW = 5, 200

# Pull data - fetch both Open and Close
data = yf.download(["QQQ", "TQQQ"], start=START, end=END, auto_adjust=True, progress=False)
opens = data["Open"].dropna()
closes = data["Close"].dropna()

# Align
idx = opens.index.intersection(closes.index)
opens = opens.loc[idx]
closes = closes.loc[idx]

print(f"Data range: {idx[0].date()} ~ {idx[-1].date()}, {len(idx)} trading days")
print()

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def event_driven_backtest(signal_src, exec_price_src, fast, slow,
                          fee_bps=2.5, slip_bps=5.0, exec_lag_days=1,
                          label=""):
    """
    Strict event-driven backtest, no look-ahead bias.

    - signal_src: which price series to compute EMA on (close)
    - exec_price_src: which price to execute at (open or close)
    - exec_lag_days: delay from signal to execution (1 = next day)
    - fee_bps: one-sided commission (basis points, 1bp = 0.01%)
    - slip_bps: slippage (basis points)
    """
    ema_f = ema(signal_src, fast)
    ema_s = ema(signal_src, slow)
    sig = (ema_f > ema_s).astype(int)
    sig.iloc[:slow] = 0  # force no signal during warm-up

    # Signal cross events (only change on the cross day)
    sig_change = sig.diff().fillna(0)  # +1 = cross up, -1 = cross down
    cross_dates = sig_change[sig_change != 0].index

    # Simulate daily equity
    cash = INIT_CASH
    shares = 0.0
    equity_curve = pd.Series(index=idx, dtype=float)
    last_action = None
    n_buy = 0
    n_sell = 0
    fee_total = 0.0
    slip_total = 0.0

    sig_change_dict = sig_change.to_dict()

    for i, t in enumerate(idx):
        # Check the signal exec_lag_days prior
        target_idx = i - exec_lag_days
        if target_idx >= 0:
            sig_t = idx[target_idx]
            change = sig_change_dict.get(sig_t, 0)
            if change == 1 and shares == 0:  # cross up, full buy
                px = exec_price_src.loc[t]
                px_with_slip = float(px) * (1 + slip_bps / 10000)
                # Buy with all cash
                shares_can_buy = cash / px_with_slip
                cost = shares_can_buy * px_with_slip
                fee = cost * fee_bps / 10000
                slip_cost = shares_can_buy * float(px) * slip_bps / 10000
                fee_total += fee
                slip_total += slip_cost
                cash -= cost + fee
                shares = shares_can_buy
                n_buy += 1
                last_action = ("BUY", t, px_with_slip, shares)
            elif change == -1 and shares > 0:  # cross down, full sell
                px = exec_price_src.loc[t]
                px_with_slip = float(px) * (1 - slip_bps / 10000)
                proceeds = shares * px_with_slip
                fee = proceeds * fee_bps / 10000
                slip_cost = shares * float(px) * slip_bps / 10000
                fee_total += fee
                slip_total += slip_cost
                cash += proceeds - fee
                shares = 0
                n_sell += 1
                last_action = ("SELL", t, px_with_slip, 0)

        # Daily equity = cash + position * day's closing price
        eq = cash + shares * float(closes.loc[t]["TQQQ" if "TQQQ" in str(exec_price_src.name) else "TQQQ"])
        equity_curve.loc[t] = eq

    # Force last day equity to use TQQQ close
    for i, t in enumerate(idx):
        equity_curve.loc[t] = cash if shares == 0 else cash + shares * float(closes["TQQQ"].loc[t])

    # Recompute equity (correction)
    cash = INIT_CASH
    shares = 0.0
    equity_curve = pd.Series(index=idx, dtype=float)
    fee_total = 0.0
    slip_total = 0.0
    n_buy = 0
    n_sell = 0

    for i, t in enumerate(idx):
        target_idx = i - exec_lag_days
        if target_idx >= 0:
            sig_t = idx[target_idx]
            change = sig_change_dict.get(sig_t, 0)
            if change == 1 and shares == 0:
                px_clean = float(exec_price_src.loc[t])
                px_with_slip = px_clean * (1 + slip_bps / 10000)
                shares_can_buy = cash / (px_with_slip * (1 + fee_bps / 10000))
                cost = shares_can_buy * px_with_slip
                fee = cost * fee_bps / 10000
                fee_total += fee
                slip_total += shares_can_buy * px_clean * slip_bps / 10000
                cash -= cost + fee
                shares = shares_can_buy
                n_buy += 1
            elif change == -1 and shares > 0:
                px_clean = float(exec_price_src.loc[t])
                px_with_slip = px_clean * (1 - slip_bps / 10000)
                proceeds = shares * px_with_slip
                fee = proceeds * fee_bps / 10000
                fee_total += fee
                slip_total += shares * px_clean * slip_bps / 10000
                cash += proceeds - fee
                shares = 0
                n_sell += 1

        eq = cash + shares * float(closes["TQQQ"].loc[t])
        equity_curve.loc[t] = eq

    # Compute metrics
    daily_ret = equity_curve.pct_change().fillna(0)
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_dd = drawdown.min()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0
    return {
        "label": label,
        "CAGR": cagr,
        "MDD": max_dd,
        "Sharpe": sharpe,
        "Calmar": calmar,
        "FinalValue": equity_curve.iloc[-1],
        "NTrades": n_buy + n_sell,
        "Fees": fee_total,
        "Slippage": slip_total,
    }


print("=" * 100)
print("Strict event-driven backtest (no look-ahead) + commission 2.5bp + slippage 5bp")
print("=" * 100)

scenarios = [
    # (label, signal_src, exec_price, exec_lag)
    ("EMA5/200 TQQQ-sig -> fill open[T+1]",  closes["TQQQ"], opens["TQQQ"], 1),
    ("EMA5/200 TQQQ-sig -> fill close[T+1]", closes["TQQQ"], closes["TQQQ"], 1),
    ("EMA5/200 QQQ-sig  -> fill open[T+1]",  closes["QQQ"],  opens["TQQQ"], 1),
    ("EMA5/200 QQQ-sig  -> fill close[T+1]", closes["QQQ"],  closes["TQQQ"], 1),
    # Dishonest control: signal = close[T], pretend filled at close[T] same day
    ("[cheating] EMA5/200 TQQQ same-day close",  closes["TQQQ"], closes["TQQQ"], 0),
]

print(f"\n{'Strategy':<48} {'CAGR':>9} {'MDD':>9} {'Sharpe':>8} {'Calmar':>8} {'Final Value':>14} {'Trades':>5}")
print("-" * 110)
results = []
for label, sig_src, exec_px, lag in scenarios:
    r = event_driven_backtest(sig_src, exec_px, FAST, SLOW, exec_lag_days=lag, label=label)
    results.append(r)
    print(f"{label:<48} {r['CAGR']*100:>8.2f}% {r['MDD']*100:>8.2f}% {r['Sharpe']:>8.3f} {r['Calmar']:>8.3f} {r['FinalValue']:>14,.0f} {r['NTrades']:>5}")

# Buy & hold control
def bh_metrics(price_series, label):
    eq = (price_series / price_series.iloc[0]) * INIT_CASH
    daily_ret = eq.pct_change().fillna(0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    max_dd = (eq / eq.cummax() - 1).min()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))
    calmar = cagr / abs(max_dd)
    return label, cagr, max_dd, sharpe, calmar, eq.iloc[-1]

print()
print("Buy & Hold control:")
print("-" * 110)
for label, c, m, s, ca, f in [bh_metrics(closes["TQQQ"], "TQQQ B&H"), bh_metrics(closes["QQQ"], "QQQ B&H")]:
    print(f"{label:<48} {c*100:>8.2f}% {m*100:>8.2f}% {s:>8.3f} {ca:>8.3f} {f:>14,.0f}     -")

print()
print("moomoo screenshot benchmark:")
print("-" * 110)
print(f"{'moomoo (new strategy 1)':<48} {47.08:>8.2f}% {-72.80:>8.2f}% {1.083:>8.3f} {0.647:>8.3f} {425895:>14,.0f}    53")

print()
print("=" * 100)
print("[Commission/Slippage Details]")
print("=" * 100)
for r in results:
    print(f"{r['label']:<48} fees ${r['Fees']:>9,.0f}  slippage ${r['Slippage']:>9,.0f}")
