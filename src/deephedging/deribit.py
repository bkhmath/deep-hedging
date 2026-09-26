"""Downloading a real option surface from Deribit, and caching it.

Crypto options for a data reason, not a modelling one: the equity index surfaces
the Heston literature uses sit behind paid vendor feeds, while Deribit publishes
marks, implied vols and the per-expiry forward on public REST endpoints.

Two facts the code has to respect. Deribit BTC options are settled in BTC while
the payoff is in USD, so a position is not the vanilla USD-payoff call that
`heston.call_price` prices; this project calibrates to the venue's quoted
implied volatilities, which are Black-76 vols on the USD payoff and so are
directly comparable. The gap is real and stated in the README. And every expiry
has its own forward (`underlying_price`), already containing the funding basis,
so each slice is calibrated in its own forward measure with rate and carry zero
- exact, not an approximation, since Heston with zero drift models the forward.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://www.deribit.com/api/v2/public"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def _get(endpoint: str, params: dict, timeout: float = 30.0) -> dict:
    """One GET against the public API, unwrapped from the {"result": ...} envelope."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    if "result" not in payload:
        raise RuntimeError(f"unexpected response from {endpoint}: {payload}")
    return payload["result"]


def download_surface(currency: str = "BTC", timeout: float = 30.0) -> pd.DataFrame:
    """Live option surface: contract terms and quotes, joined on instrument name."""
    instruments = _get(
        "get_instruments", {"currency": currency, "kind": "option", "expired": "false"}, timeout
    )
    quotes = _get("get_book_summary_by_currency", {"currency": currency, "kind": "option"}, timeout)

    terms = pd.DataFrame(instruments)[
        ["instrument_name", "strike", "option_type", "expiration_timestamp"]
    ]
    book = pd.DataFrame(quotes)[
        [
            "instrument_name",
            "mark_price",
            "mark_iv",
            "underlying_price",
            "bid_price",
            "ask_price",
            "open_interest",
            "volume",
        ]
    ]
    surface = terms.merge(book, on="instrument_name", how="inner")
    surface["snapshot_ms"] = int(time.time() * 1000)
    return surface


def cache_path(currency: str = "BTC", stamp: str | None = None) -> Path:
    stamp = stamp or time.strftime("%Y%m%d")
    return CACHE_DIR / f"deribit_{currency.lower()}_surface_{stamp}.parquet"


def save_surface(surface: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    surface.to_parquet(path, index=False)
    return path


def latest_cached(currency: str = "BTC") -> Path | None:
    """Most recent cached snapshot, or None. The cache is what makes the study
    runnable offline."""
    if not CACHE_DIR.exists():
        return None
    matches = sorted(CACHE_DIR.glob(f"deribit_{currency.lower()}_surface_*.parquet"))
    return matches[-1] if matches else None


def load_surface(path: Path | None = None, currency: str = "BTC") -> pd.DataFrame:
    path = path or latest_cached(currency)
    if path is None:
        raise FileNotFoundError(
            "no cached Deribit surface found; run `python scripts/download_surface.py` "
            "or pass --synthetic to run the study offline"
        )
    return pd.read_parquet(path)


def clean_surface(
    surface: pd.DataFrame,
    min_days: float = 5.0,
    max_days: float = 200.0,
    min_moneyness: float = 0.6,
    max_moneyness: float = 1.6,
    min_open_interest: float = 1.0,
) -> pd.DataFrame:
    """Filter the raw snapshot down to quotes worth fitting.

    Maturity 5 to 200 days: shorter and the implied vol is dominated by the tick
    size of a nearly worthless option, longer and the open interest thins out.
    Moneyness K/F in [0.6, 1.6], outside which the mark price is a few ticks.
    Out-of-the-money only (puts below the forward, calls above), which are the
    liquid quotes and lose nothing, since put-call parity gives the in-the-money
    option the same implied vol at the same strike. Non-zero open interest and a
    positive two-sided quote, so a stale instrument stays out of the objective.

    Adds: maturity in years, forward, moneyness, market implied vol as a
    decimal, and a relative spread used as the fitting weight.
    """
    df = surface.copy()
    now_ms = float(df["snapshot_ms"].iloc[0])
    df["maturity"] = (df["expiration_timestamp"] - now_ms) / 1000.0 / SECONDS_PER_YEAR
    df["forward"] = df["underlying_price"]
    df["moneyness"] = df["strike"] / df["forward"]
    df["market_iv"] = df["mark_iv"] / 100.0  # the venue quotes it in percent

    days = df["maturity"] * 365.0
    otm = ((df["option_type"] == "put") & (df["moneyness"] < 1.0)) | (
        (df["option_type"] == "call") & (df["moneyness"] >= 1.0)
    )
    keep = (
        days.between(min_days, max_days)
        & df["moneyness"].between(min_moneyness, max_moneyness)
        & otm
        & (df["open_interest"] >= min_open_interest)
        & (df["bid_price"] > 0)
        & (df["ask_price"] > 0)
        & (df["market_iv"] > 0.01)
        & np.isfinite(df["market_iv"])
    )
    df = df.loc[keep].copy()

    # Inverse weight: a quote the venue itself quotes loosely should not drag
    # the fit.
    mid = 0.5 * (df["bid_price"] + df["ask_price"])
    df["rel_spread"] = (df["ask_price"] - df["bid_price"]) / mid.clip(lower=1e-12)
    return df.sort_values(["maturity", "strike"]).reset_index(drop=True)
