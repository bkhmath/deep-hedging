"""Download and cache one snapshot of the Deribit BTC option surface.

Run this once before `make study`. The study reads the cache, never the network,
so a result can be reproduced after the quotes have moved on, which for a crypto
venue is a matter of seconds.

    python scripts/download_surface.py
"""

from __future__ import annotations

import argparse

from deephedging import deribit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currency", default="BTC")
    args = parser.parse_args()

    print(f"fetching the live {args.currency} option surface from Deribit ...")
    raw = deribit.download_surface(args.currency)
    path = deribit.save_surface(raw, deribit.cache_path(args.currency))
    clean = deribit.clean_surface(raw)

    print(f"  {len(raw)} instruments quoted, {len(clean)} usable after filtering")
    print(f"  forward {clean['forward'].iloc[0]:,.0f} USD")
    print(
        f"  maturities {clean['maturity'].min() * 365:.0f} to {clean['maturity'].max() * 365:.0f} days"
    )
    print(f"  implied vols {clean['market_iv'].min():.1%} to {clean['market_iv'].max():.1%}")
    print(f"cached to {path}")


if __name__ == "__main__":
    main()
