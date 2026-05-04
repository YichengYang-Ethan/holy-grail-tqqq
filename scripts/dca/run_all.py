"""
One-shot runner: end-to-end DCA Holy Grail backtest.

Runs in this order:
  1. retail_holy_grail.main()  → cross-section by r
  2. baselines.main()          → 8 strategies vs Retail
  3. cpcv_dca.main()           → rolling 5/10/15 yr + fold CPCV + WF OOS
  4. peer_review_dca.main()    → 7 robustness checks
  5. figures_dca.main()        → 4 PNGs

Each step is independent (cached pickles in results/dca/) so steps can be
re-run individually without restarting the universe.
"""
from __future__ import annotations
import argparse
import sys
import time


def _print_banner(s: str) -> None:
    print()
    print("█" * 100)
    print(f"█  {s}")
    print("█" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[],
                     choices=["retail", "baselines", "cpcv", "peer_review", "figures"],
                     help="Steps to skip.")
    args = ap.parse_args()

    t0 = time.time()

    if "retail" not in args.skip:
        _print_banner("[1/5] Retail Holy Grail — cross-section by r")
        from . import retail_holy_grail
        retail_holy_grail.main()

    if "baselines" not in args.skip:
        _print_banner("[2/5] Baselines — 8 strategies under same cashflow")
        from . import baselines
        baselines.main()

    if "cpcv" not in args.skip:
        _print_banner("[3/5] CPCV / Walk-forward — rolling 5/10/15 + fold + OOS")
        from . import cpcv_dca
        cpcv_dca.main()

    if "peer_review" not in args.skip:
        _print_banner("[4/5] Peer Review — 7 robustness checks")
        from . import peer_review_dca
        peer_review_dca.main()

    if "figures" not in args.skip:
        _print_banner("[5/5] Figures — 4 PNGs to figures/dca/")
        from . import figures_dca
        figures_dca.main()

    dt = time.time() - t0
    _print_banner(f"DONE — total elapsed {dt:.1f}s "
                   f"(skipped: {args.skip if args.skip else 'none'})")


if __name__ == "__main__":
    main()
