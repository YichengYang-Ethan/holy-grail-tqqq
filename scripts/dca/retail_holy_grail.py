"""
Retail Holy Grail — main entry for the DCA-augmented EMA(5,200) rotation.

Encapsulates the r-rule (initial / annual_contribution):
    r ≤ 2  → 100% LSI day-0,            no phase-in
    2 < r ≤ 5 → 70% LSI day-0 + 30% over 6 months
    r > 5  → 50% LSI day-0 + 50% over 6 months

Plus mechanical signal-following on monthly contributions.

Outputs:
  - Single-path metrics for a given (initial, annual) profile.
  - Cross-section table over r ∈ {1, 3, 7} so different retail profiles can be
    looked up directly.
"""
from __future__ import annotations
import argparse
import pandas as pd

from .core import (
    load_universe, signal_ema,
    make_retail_cashflow, make_lump_cashflow, make_pure_dca_cashflow,
    backtest_dca, metrics_dca, fmt_metric_row,
)


def run_retail(initial: float, annual: float, U: dict | None = None,
               fast: int = 5, slow: int = 200,
               contrib_freq: str = "M") -> dict:
    if U is None:
        U = load_universe()
    sig = signal_ema(U["qqq"], fast, slow)
    cf = make_retail_cashflow(initial, annual, U["qqq"].index,
                               contrib_freq=contrib_freq)
    res = backtest_dca(U["tqqq"], U["qqq"], sig, cf,
                        asset_on_name="TQQQ", asset_off_name="QQQ")
    return {"result": res, "metrics": metrics_dca(res), "cashflow": cf}


def cross_section(profiles: list[tuple[float, float]],
                  U: dict | None = None) -> pd.DataFrame:
    """Run the retail strategy across multiple (initial, annual) profiles."""
    if U is None:
        U = load_universe()
    rows = []
    for init, ann in profiles:
        m = run_retail(init, ann, U)["metrics"]
        m["initial"] = init
        m["annual"] = ann
        m["r"] = init / ann if ann > 0 else float("inf")
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["initial", "annual", "r", "label", "total_contributed", "final_value",
            "multiple", "xirr", "twr", "mdd_market", "mdd_vs_contrib",
            "sharpe", "sortino", "calmar_xirr", "n_rotations",
            "trough_date", "trough_equity", "trough_contributed",
            "trough_dd_market", "trough_dd_vs_contrib"]
    return df[[c for c in cols if c in df.columns]]


def print_cross_section(df: pd.DataFrame) -> None:
    print()
    print("=" * 130)
    print(f"{'Profile (init / annual)':<28}{'r':>5}{'XIRR':>10}{'TWR':>9}"
          f"{'MDD-mkt':>10}{'MDD-contrib':>13}{'Mult':>9}{'Final':>16}{'Total in':>14}{'Rot':>5}")
    print("-" * 130)
    for _, r in df.iterrows():
        prof = f"${r['initial']:>5,.0f} + ${r['annual']:>5,.0f}/yr"
        print(f"{prof:<28}{r['r']:>5.1f}{r['xirr']*100:>9.2f}%{r['twr']*100:>8.2f}%"
              f"{r['mdd_market']*100:>9.2f}%{r['mdd_vs_contrib']*100:>12.2f}%"
              f"{r['multiple']:>8.2f}x${r['final_value']:>14,.0f}"
              f"${r['total_contributed']:>12,.0f}{r['n_rotations']:>5}")
    print("=" * 130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial", type=float, default=10_000)
    ap.add_argument("--annual", type=float, default=10_000)
    ap.add_argument("--freq", choices=["M", "Q", "Y"], default="M")
    ap.add_argument("--fast", type=int, default=5)
    ap.add_argument("--slow", type=int, default=200)
    args = ap.parse_args()

    print("[retail_holy_grail] Loading universe ...")
    U = load_universe()
    print(f"  QQQ {U['qqq'].index[0].date()} → {U['qqq'].index[-1].date()}"
          f" ({len(U['qqq'])} days)")
    print()

    print("=" * 130)
    print("[Single profile] Retail Holy Grail backtest")
    print("=" * 130)
    out = run_retail(args.initial, args.annual, U, args.fast, args.slow, args.freq)
    print(out["cashflow"].label)
    print(fmt_metric_row(out["metrics"]))
    m = out["metrics"]
    print(f"  Trough: {m['trough_date'].date()} | equity ${m['trough_equity']:,.0f}"
          f" | contributed ${m['trough_contributed']:,.0f}"
          f" | DD-vs-contrib {m['trough_dd_vs_contrib']*100:.1f}%")

    # Cross-section: classic retail profiles
    print()
    print("=" * 130)
    print("[Cross-section] Retail profiles by r-bucket")
    print("=" * 130)
    profiles = [
        (5_000, 10_000),    # r=0.5 - young grad, small starter
        (10_000, 10_000),   # r=1.0 - typical young saver
        (10_000, 5_000),    # r=2.0 - boundary of 100%-LSI bucket
        (30_000, 10_000),   # r=3.0 - mid-career
        (50_000, 10_000),   # r=5.0 - boundary of 70%-LSI bucket
        (100_000, 10_000),  # r=10.0 - already-accumulated
        (10_000, 0),        # r=inf - lump only (matches blog baseline)
    ]
    df = cross_section(profiles, U)
    print_cross_section(df)

    return df


if __name__ == "__main__":
    main()
