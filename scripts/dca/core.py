"""
DCA Holy Grail: Core engine for retail-version backtest.

Reuses synthetic TQQQ + T-bill proxy from peer_review_fixes.py / synth_tqqq_v2.py
but replaces the lump-sum return-compounding loop with a cashflow + share-tracking
loop suitable for DCA (Dollar-Cost Averaging) scenarios.

Conventions:
- B-case integer rebalancing: on every signal flip OR cash-injection, the entire
  portfolio is rotated to the asset implied by the *current* signal.
- Two-asset universe: risk-on (TQQQ) vs risk-off (QQQ or T-bill).
- Costs: fee_bps + slip_bps on rotated dollar-volume (one-side).
- DCA-aware metrics: XIRR (money-weighted), TWR (time-weighted),
  drawdown vs market value, drawdown vs cumulative contributions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Literal, Optional
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# Data layer (mirrors peer_review_fixes.build_synth_tqqq + build_tbill_proxy)
# --------------------------------------------------------------------------

START = "1999-03-10"
END = "2026-04-18"


def load_prices(start: str = START, end: str = END,
                include_spy: bool = True) -> dict:
    """Download adjusted close series for QQQ, TQQQ, optionally SPY + AGG."""
    out = {}
    out["qqq"] = yf.download("QQQ", start=start, end=end,
                              auto_adjust=True, progress=False)["Close"].squeeze()
    out["tqqq_real"] = yf.download("TQQQ", start="2010-02-11", end=end,
                                    auto_adjust=True, progress=False)["Close"].squeeze()
    if include_spy:
        out["spy"] = yf.download("SPY", start=start, end=end,
                                  auto_adjust=True, progress=False)["Close"].squeeze()
        out["agg"] = yf.download("AGG", start="2003-09-29", end=end,
                                  auto_adjust=True, progress=False)["Close"].squeeze()
    return out


def build_synth_tqqq(qqq: pd.Series, tqqq_real: pd.Series) -> pd.Series:
    """Pre-2010 synthetic TQQQ from QQQ daily returns, calibrated to real
    TQQQ at the splice date (2010-02-11). Identical model as peer_review_fixes.py."""
    qret = qqq.pct_change().fillna(0)
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
    fin_d = pd.Series([fin(d) for d in qqq.index], index=qqq.index)
    slip_d = 0.003 / 252
    synth_ret = 3 * qret - expense_d - fin_d - slip_d
    synth = (1 + synth_ret).cumprod()
    overlap = tqqq_real.index[0]
    calib = float(tqqq_real.iloc[0]) / float(synth.loc[overlap])
    pre = synth.loc[:overlap].iloc[:-1] * calib
    full = pd.concat([pre, tqqq_real])
    return full.reindex(qqq.index).ffill()


def build_tbill_proxy(idx: pd.DatetimeIndex) -> pd.Series:
    """Piecewise-constant T-bill cumulative price, base=100."""
    rates = []
    for d in idx:
        y = d.year
        if y <= 2000: r = 0.055
        elif y <= 2007: r = 0.040
        elif y <= 2008: r = 0.020
        elif y <= 2015: r = 0.0010
        elif y <= 2019: r = 0.015
        elif y <= 2021: r = 0.0005
        elif y <= 2022: r = 0.02
        else: r = 0.045
        rates.append(r / 252)
    return pd.Series(np.cumprod(1 + np.array(rates)), index=idx) * 100


# --------------------------------------------------------------------------
# Signal generators (mirrors peer_review_fixes signal_*)
# --------------------------------------------------------------------------

def signal_ema(close: pd.Series, fast: int, slow: int) -> pd.Series:
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    sig = (f > s).astype(float)
    sig.iloc[:slow] = 0.0
    return sig


def signal_sma(close: pd.Series, fast: int, slow: int) -> pd.Series:
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    sig = (f > s).astype(float)
    sig.iloc[:slow] = 0.0
    return sig


def signal_price_vs_sma(close: pd.Series, length: int) -> pd.Series:
    ma = close.rolling(length).mean()
    sig = (close > ma).astype(float)
    sig.iloc[:length] = 0.0
    return sig


# --------------------------------------------------------------------------
# Cashflow plan
# --------------------------------------------------------------------------

@dataclass
class CashflowPlan:
    """A schedule of cash injections into the strategy account.

    Attributes
    ----------
    injections : pd.Series indexed by trading-day timestamps (date of injection),
        values are dollars to add. The first injection is the initial deposit.
    label : str — for reporting.
    """
    injections: pd.Series
    label: str = ""

    def total(self) -> float:
        return float(self.injections.sum())


def _first_business_day_each_month(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each (year, month) in idx, return the first index entry."""
    df = pd.DataFrame({"d": idx}, index=idx)
    return df.groupby([idx.year, idx.month]).first()["d"].values


def make_retail_cashflow(initial: float,
                         annual_contribution: float,
                         dates: pd.DatetimeIndex,
                         start_date: Optional[str] = None,
                         contrib_freq: Literal["M", "Q", "Y"] = "M") -> CashflowPlan:
    """Apply the r-rule (r = initial / annual_contribution):
        r ≤ 2  → 100% LSI on day 0, no phase-in
        2 < r ≤ 5  → 70% LSI day 0 + 30% over 6 months
        r > 5  → 50% LSI day 0 + 50% over 6 months
    The annual_contribution is split evenly across the chosen frequency
    (month/quarter/year) starting *after* phase-in is complete.
    """
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if len(dates) == 0:
        raise ValueError("No trading dates available after start_date.")

    if annual_contribution > 0:
        r = initial / annual_contribution
    else:
        r = float("inf")

    if r <= 2:
        lsi_pct, phase_months = 1.0, 0
    elif r <= 5:
        lsi_pct, phase_months = 0.7, 6
    else:
        lsi_pct, phase_months = 0.5, 6

    # First business day per month within the supplied range
    fbom = pd.to_datetime(_first_business_day_each_month(dates))
    fbom = pd.DatetimeIndex(fbom).sort_values()

    injections = pd.Series(0.0, index=dates, dtype=float)

    # Initial LSI
    injections.iloc[0] += initial * lsi_pct

    # Phase-in DCA (monthly chunks starting from month 1 fbom)
    if phase_months > 0:
        chunk = initial * (1 - lsi_pct) / phase_months
        for i in range(1, phase_months + 1):
            if i < len(fbom):
                injections.loc[fbom[i]] += chunk

    # Recurring contributions (after phase-in complete)
    if annual_contribution > 0:
        if contrib_freq == "M":
            n_per_year = 12
            chunk_dates = fbom
        elif contrib_freq == "Q":
            n_per_year = 4
            chunk_dates = fbom[::3]
        elif contrib_freq == "Y":
            n_per_year = 1
            chunk_dates = fbom[::12]
        else:
            raise ValueError(contrib_freq)
        chunk = annual_contribution / n_per_year
        skip = phase_months if contrib_freq == "M" else (
            (phase_months // 3 + (phase_months % 3 > 0)) if contrib_freq == "Q"
            else (1 if phase_months > 0 else 0))
        for d in chunk_dates[skip + 1:]:  # +1 to skip the day-0 month
            if d in injections.index:
                injections.loc[d] += chunk

    label = (f"Retail r={r:.1f} → LSI={lsi_pct*100:.0f}% + {phase_months}mo phase-in"
             f" + ${annual_contribution:,.0f}/yr ({contrib_freq})")
    return CashflowPlan(injections=injections, label=label)


def make_lump_cashflow(initial: float, dates: pd.DatetimeIndex,
                       start_date: Optional[str] = None) -> CashflowPlan:
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    inj = pd.Series(0.0, index=dates, dtype=float)
    inj.iloc[0] = initial
    return CashflowPlan(inj, f"Lump-sum ${initial:,.0f}")


def make_pure_dca_cashflow(initial: float, annual_contribution: float,
                            dates: pd.DatetimeIndex,
                            start_date: Optional[str] = None,
                            contrib_freq: Literal["M", "Q", "Y"] = "M") -> CashflowPlan:
    """Pure DCA: 0% LSI, every dollar (incl. initial) flows in monthly over
    the first 12 months, then recurring contributions."""
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    fbom = pd.DatetimeIndex(pd.to_datetime(_first_business_day_each_month(dates)))
    inj = pd.Series(0.0, index=dates, dtype=float)
    chunk_init = initial / 12
    for i in range(min(12, len(fbom))):
        inj.loc[fbom[i]] += chunk_init
    if annual_contribution > 0:
        if contrib_freq == "M":
            chunk = annual_contribution / 12
            for d in fbom[12:]:
                if d in inj.index:
                    inj.loc[d] += chunk
    return CashflowPlan(inj, f"Pure DCA ${initial:,.0f} + ${annual_contribution:,.0f}/yr")


# --------------------------------------------------------------------------
# DCA backtest engine (B-case full rotation)
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity: pd.Series                  # market value over time
    contributions_cum: pd.Series       # cumulative cash injected
    held_asset: pd.Series              # 'on' / 'off' / 'none' per day
    n_rotations: int
    cashflow: CashflowPlan
    asset_on_name: str
    asset_off_name: str
    fee_bps: float
    slip_bps: float


def backtest_dca(price_on: pd.Series,
                 price_off: pd.Series,
                 signal: pd.Series,
                 cashflow: CashflowPlan,
                 fee_bps: float = 2.5,
                 slip_bps: float = 5.0,
                 asset_on_name: str = "TQQQ",
                 asset_off_name: str = "QQQ") -> BacktestResult:
    """B-case integer rotation backtest with cash injections.

    Implementation strategy: piggyback on the lump-sum daily-return-compounding
    model from holy-grail's `backtest_dual` (so rotation costs and rotation-day
    new-asset returns are handled identically), then *layer in cash injections*
    as same-day deposits paying one-side buy-cost only.

      qpos[t] = (1 - signal[t-1])   # weight in OFF asset (QQQ)
      tpos[t] = signal[t-1]         # weight in ON asset (TQQQ)
      strat_ret[t] = qpos*q_ret + tpos*t_ret − rotation_cost
      equity[t] = equity[t-1] * (1 + strat_ret[t])
                  + injection[t] * (1 − buy_cost_if_invested)

    Convention: signal at the close of day t-1 dictates the position for day t,
    so on day t the strategy already earns the new asset's return (matches the
    original `pos.shift(1)` behavior). On day 0 weights are 0 (no return), and
    the day-0 injection is held as cash; in warmup (t<slow) qpos defaults to 1
    so the strategy holds OFF (QQQ).
    """
    idx = price_on.index.intersection(price_off.index).intersection(signal.index)
    idx = idx.intersection(cashflow.injections.index).sort_values()
    p_on = price_on.reindex(idx)
    p_off = price_off.reindex(idx)
    sig = signal.reindex(idx).fillna(0.0).astype(float)
    inj = cashflow.injections.reindex(idx).fillna(0.0).astype(float)

    bull = sig
    # holy-grail's `rotation_fixed`: qpos = 1 - bull, tpos = bull, then shift(1)
    qpos_lag = (1.0 - bull).shift(1).fillna(0.0)
    tpos_lag = bull.shift(1).fillna(0.0)
    q_ret = p_off.pct_change().fillna(0.0)
    t_ret = p_on.pct_change().fillna(0.0)
    cost_bps = (fee_bps + slip_bps) / 10000.0
    rot_cost = (qpos_lag.diff().abs().fillna(0.0)
                + tpos_lag.diff().abs().fillna(0.0)) * cost_bps
    strat_ret = qpos_lag * q_ret + tpos_lag * t_ret - rot_cost

    n = len(idx)
    eq = np.zeros(n)
    held = np.empty(n, dtype=object)

    qp = qpos_lag.values
    tp = tpos_lag.values
    sr = strat_ret.values
    ij = inj.values

    for t in range(n):
        if t == 0:
            # day 0: no prior position, no return; injection held (no signal yet)
            eq[0] = ij[0]
        else:
            base = eq[t-1] * (1.0 + sr[t])
            if ij[t] > 0:
                # one-side buy cost if today the strategy is invested
                buy_cost = cost_bps if (qp[t] > 0 or tp[t] > 0) else 0.0
                base += ij[t] * (1.0 - buy_cost)
            eq[t] = base
        if tp[t] > 0:
            held[t] = "on"
        elif qp[t] > 0:
            held[t] = "off"
        else:
            held[t] = "none"

    n_rot = int(((qpos_lag.diff().abs() + tpos_lag.diff().abs()) > 0.5).sum())

    return BacktestResult(
        equity=pd.Series(eq, index=idx),
        contributions_cum=inj.cumsum(),
        held_asset=pd.Series(held, index=idx),
        n_rotations=n_rot,
        cashflow=cashflow,
        asset_on_name=asset_on_name,
        asset_off_name=asset_off_name,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
    )


# --------------------------------------------------------------------------
# DCA-aware metrics
# --------------------------------------------------------------------------

def xirr(injections: pd.Series, final_value: float, final_date: pd.Timestamp) -> float:
    """Money-weighted return (XIRR). injections are positive cash *outflows*
    from the investor's perspective; final_value is a single positive *inflow*
    on final_date.
    Solves: Σ cf_i / (1+r)^(t_i / 365) = 0  with cf signs (- for invested, + final)."""
    cf_dates = list(injections.index[injections != 0]) + [final_date]
    cf_amounts = list(-injections[injections != 0].values) + [final_value]
    if len(cf_dates) < 2 or final_value <= 0:
        return float("nan")
    t0 = cf_dates[0]
    days = np.array([(d - t0).days for d in cf_dates], dtype=float)
    amounts = np.array(cf_amounts, dtype=float)

    def npv(r):
        return float(np.sum(amounts / (1.0 + r) ** (days / 365.25)))

    # bracket
    try:
        lo, hi = -0.99, 10.0
        if npv(lo) * npv(hi) > 0:
            return float("nan")
        return brentq(npv, lo, hi, maxiter=200)
    except Exception:
        return float("nan")


def time_weighted_return(equity: pd.Series, injections: pd.Series) -> float:
    """Annualized TWR — chains daily sub-period returns that exclude cashflow.
    Sub-period return on day t = (eq[t] - inj[t]) / eq[t-1] - 1."""
    eq = equity.values
    inj = injections.reindex(equity.index).fillna(0.0).values
    rets = []
    for t in range(1, len(eq)):
        if eq[t-1] <= 0:
            continue
        r = (eq[t] - inj[t]) / eq[t-1] - 1.0
        rets.append(r)
    rets = np.array(rets)
    cum = float(np.prod(1.0 + rets))
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or cum <= 0:
        return float("nan")
    return cum ** (1.0 / years) - 1.0


def dd_vs_market(equity: pd.Series) -> float:
    if equity.empty: return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def dd_vs_contributions(equity: pd.Series, contrib_cum: pd.Series) -> float:
    """How far did market value fall below 'cash you've put in' at each point?
    Returns max value of (contrib_cum - equity) / contrib_cum, with sign convention
    matching dd_vs_market (negative = worse)."""
    common = equity.index.intersection(contrib_cum.index)
    eq = equity.loc[common]
    cc = contrib_cum.loc[common]
    mask = cc > 0
    if mask.sum() == 0: return 0.0
    ratio = (eq[mask] - cc[mask]) / cc[mask]
    return float(ratio.min())


def trough_summary(equity: pd.Series, contrib_cum: pd.Series) -> dict:
    """At market-value DD trough: how much had been contributed, what was MV?"""
    if equity.empty: return {}
    dd = equity / equity.cummax() - 1.0
    trough_t = dd.idxmin()
    return {
        "trough_date": trough_t,
        "trough_equity": float(equity.loc[trough_t]),
        "trough_contributed": float(contrib_cum.loc[trough_t]),
        "trough_dd_market": float(dd.loc[trough_t]),
        "trough_dd_vs_contrib": float(
            (equity.loc[trough_t] - contrib_cum.loc[trough_t]) / contrib_cum.loc[trough_t]
            if contrib_cum.loc[trough_t] > 0 else 0.0),
    }


def metrics_dca(result: BacktestResult) -> dict:
    eq = result.equity
    inj = result.cashflow.injections
    cc = result.contributions_cum
    final_val = float(eq.iloc[-1])
    total_in = float(inj.sum())
    multiple = final_val / total_in if total_in > 0 else float("nan")

    # XIRR
    xirr_v = xirr(inj, final_val, eq.index[-1])
    twr_v = time_weighted_return(eq, inj)

    # DDs
    mdd_mkt = dd_vs_market(eq)
    mdd_con = dd_vs_contributions(eq, cc)
    trough = trough_summary(eq, cc)

    # Sharpe / Sortino on TWR-style sub-period returns
    eq_arr = eq.values
    inj_arr = inj.reindex(eq.index).fillna(0.0).values
    rets = []
    for t in range(1, len(eq_arr)):
        if eq_arr[t-1] <= 0: continue
        rets.append((eq_arr[t] - inj_arr[t]) / eq_arr[t-1] - 1.0)
    rets = np.array(rets)
    if len(rets) > 1 and rets.std() > 0:
        sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
    else:
        sharpe = float("nan")
    neg = rets[rets < 0]
    sortino = (rets.mean() * 252) / (neg.std() * np.sqrt(252)) if len(neg) > 1 and neg.std() > 0 else float("nan")
    calmar = xirr_v / abs(mdd_mkt) if (mdd_mkt < 0 and not np.isnan(xirr_v)) else float("nan")

    return {
        "label": result.cashflow.label,
        "total_contributed": total_in,
        "final_value": final_val,
        "multiple": multiple,
        "xirr": xirr_v,
        "twr": twr_v,
        "mdd_market": mdd_mkt,
        "mdd_vs_contrib": mdd_con,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar_xirr": calmar,
        "n_rotations": result.n_rotations,
        **trough,
    }


def fmt_metric_row(m: dict, name_width: int = 50) -> str:
    """Format one metrics dict as a single text line."""
    lbl = m.get("label", "")
    return (f"{lbl:<{name_width}}"
            f" XIRR {m['xirr']*100:>6.2f}%"
            f" TWR {m['twr']*100:>6.2f}%"
            f" MDD-mkt {m['mdd_market']*100:>7.2f}%"
            f" MDD-vs-contrib {m['mdd_vs_contrib']*100:>7.2f}%"
            f" Final ${m['final_value']:>14,.0f}"
            f" Mult {m['multiple']:>6.2f}x"
            f" Rot {m['n_rotations']:>3}")


# --------------------------------------------------------------------------
# Convenience: end-to-end loader
# --------------------------------------------------------------------------

def load_universe(start: str = START, end: str = END) -> dict:
    """One-call: download QQQ/TQQQ/SPY/AGG, splice synthetic TQQQ, build T-bill."""
    px = load_prices(start, end, include_spy=True)
    tqqq_full = build_synth_tqqq(px["qqq"], px["tqqq_real"])
    tbill = build_tbill_proxy(px["qqq"].index)
    # Align AGG to QQQ index for 60/40 baseline
    agg_aligned = px["agg"].reindex(px["qqq"].index).ffill()
    spy_aligned = px["spy"].reindex(px["qqq"].index).ffill()
    return {
        "qqq": px["qqq"],
        "tqqq": tqqq_full,
        "tqqq_real": px["tqqq_real"],
        "spy": spy_aligned,
        "agg": agg_aligned,
        "tbill": tbill,
    }


if __name__ == "__main__":
    import sys
    print("Loading universe ...")
    U = load_universe()
    print(f"QQQ {U['qqq'].index[0].date()} → {U['qqq'].index[-1].date()} ({len(U['qqq'])} days)")
    print(f"TQQQ (synth+real) coverage matches QQQ: {U['tqqq'].index.equals(U['qqq'].index)}")

    # Smoke test
    sig = signal_ema(U["qqq"], 5, 200)
    cf = make_retail_cashflow(initial=10_000, annual_contribution=10_000,
                               dates=U["qqq"].index, contrib_freq="M")
    print(f"\nCashflow plan: {cf.label}")
    print(f"  Total to be contributed: ${cf.total():,.0f}")
    print(f"  Number of injection events: {(cf.injections > 0).sum()}")

    res = backtest_dca(U["tqqq"], U["qqq"], sig, cf,
                        asset_on_name="TQQQ", asset_off_name="QQQ")
    m = metrics_dca(res)
    print()
    print(fmt_metric_row(m))
    print(f"Trough: {m.get('trough_date')} eq=${m.get('trough_equity'):,.0f}, "
          f"contributed=${m.get('trough_contributed'):,.0f}")
