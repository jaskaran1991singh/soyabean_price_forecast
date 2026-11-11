"""Global prices and exchange rates.

  yfinance  daily closes: CBOT soybeans / soymeal / soy oil futures (front month), Brent crude,
            USD/INR, USD/BRL, US dollar index. Tickers come from settings.markets.yfinance.
  FRED      monthly IMF commodity prices (palm oil, soy oil, soybeans, soymeal, sunflower oil),
            downloaded from FRED's public CSV endpoint (no key). Used mainly for palm oil,
            which has no free daily feed.

Output table `markets_daily` (long format, one row per date per series):
  obs_date, series (our name, e.g. "cbot_soybean"), value, source, available_at
Units are as quoted: CBOT soybeans in US cents/bushel, meal in USD/short ton, oil in US cents/lb,
FRED series in USD/tonne. Conversion to Rs/quintal happens in features/build.py.
"""
from __future__ import annotations

import io
import logging

import pandas as pd

from .. import store
from ..config import Settings
from .http import get

log = logging.getLogger(__name__)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def collect_yf(s: Settings) -> pd.DataFrame:
    """Daily closing prices for each configured Yahoo Finance ticker.

    Example:
        collect_yf(s) -> long DataFrame[obs_date, series, value, source="yfinance", available_at]
                         e.g. rows (2024-11-01, "cbot_soybean", 1012.5), (2024-11-01, "usdinr", 84.1), ...
    """
    import yfinance as yf  # imported here so the rest of the pipeline works without yfinance

    tickers = s["markets"]["yfinance"]
    frames = []
    for name, tk in tickers.items():
        try:
            # auto_adjust=False: we want the raw quoted close, not a dividend-adjusted series.
            h = yf.download(tk, start=s.start_date, end=s.end_date, progress=False, auto_adjust=False)
        except Exception as e:  # noqa: BLE001 - one bad ticker shouldn't stop the others
            log.warning("yfinance %s failed: %s", tk, e)
            continue
        if h is None or h.empty:
            log.warning("yfinance %s returned no data", tk)
            continue
        close = h["Close"]
        # Newer yfinance returns a one-column DataFrame (multi-index columns); take it as a Series.
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        df = close.dropna().rename("value").reset_index().rename(columns={"Date": "obs_date"})
        df["series"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["obs_date"] = pd.to_datetime(out["obs_date"]).dt.tz_localize(None).dt.normalize()
    out["source"] = "yfinance"
    # US markets close after Indian market hours, so a US close is usable from the next day.
    out["available_at"] = out["obs_date"] + pd.Timedelta(days=1)
    return out


def collect_fred(s: Settings) -> pd.DataFrame:
    """Monthly FRED series. A month's average is only published after the month ends, so
    available_at = last day of that month + fred_publication_lag_days (35 by default).

    Example:
        collect_fred(s) -> long DataFrame[obs_date, series, value, source="fred", available_at]
                           e.g. (2024-10-01, "palm_oil_usd_t", 1065.0, available_at=2024-12-05)
    """
    frames = []
    lag = pd.Timedelta(days=s["markets"].get("fred_publication_lag_days", 35))
    for name, sid in s["markets"].get("fred_monthly", {}).items():
        try:
            txt = get(FRED_CSV, params={"id": sid}).text
        except Exception as e:  # noqa: BLE001
            log.warning("FRED %s failed: %s", sid, e)
            continue
        df = pd.read_csv(io.StringIO(txt))
        df.columns = ["obs_date", "value"]                        # CSV is: observation_date,<SERIES_ID>
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED writes "." for missing
        df = df.dropna()
        df["obs_date"] = pd.to_datetime(df["obs_date"])            # first day of the month
        df["series"] = name
        df["available_at"] = df["obs_date"] + pd.offsets.MonthEnd(0) + lag
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "fred"
    # Keep one extra year before start_date so 4-week/annual changes exist from the first week.
    return out[out["obs_date"] >= pd.Timestamp(s.start_date) - pd.DateOffset(years=1)]


def collect(s: Settings) -> pd.DataFrame:
    """Collect both sources and merge into data/staged/markets_daily.parquet.

    Example:
        collect(s) -> long DataFrame[obs_date, series, value, source, available_at] for every yfinance
                      + FRED series, and writes data/staged/markets_daily.parquet
    """
    frames = [collect_yf(s), collect_fred(s)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return store.read(s, "markets_daily")
    return store.upsert(s, "markets_daily", pd.concat(frames, ignore_index=True), ["obs_date", "series"])