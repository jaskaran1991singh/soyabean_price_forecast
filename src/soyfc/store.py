"""Layered storage for the pipeline — plain functions, each taking the Settings object.

    data/<target>/raw/<source>/<hash>.json|zip   exact API / search responses (cache, never edited)
    data/<target>/staged/<table>.parquet         cleaned tables, one per source  <- this module
    data/<target>/features/...                   final training outputs (written by features/build.py)

`<target>` is the configured market's folder (config.Settings.work_dir, e.g. `indore_soyabean`),
so each mandi's data is isolated and switching markets never corrupts another's staged tables.

There is no Store object: collectors, the agent and the feature builder all just
`from .. import store` and call e.g. `store.read(s, "mandi_daily")`.

Point-in-time rule: every staged table carries two time columns
    obs_date      the date the value describes        (e.g. the mandi trading day)
    available_at  the earliest date it could be known (e.g. trading day + 1, article publish date)
The feature builder only uses rows whose available_at <= the forecast date, so the
training data never contains information from the future.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import Settings

log = logging.getLogger(__name__)


# =====================================================================
# Raw cache — stores the untouched response of each request so re-runs
# never pay twice for the same Tavily search or re-download old data.
# =====================================================================
def _raw_path(s: Settings, source: str, key: str, ext: str) -> Path:
    """File path for one cached response.

    source : sub-folder per data source, e.g. "tavily", "openmeteo", "datagov"
    key    : string that uniquely describes the request (URL + parameters)
    ext    : file extension ("json", "zip", ...)
    The key can be long and contain characters Windows forbids in filenames, so the filename is a
    short SHA-1 hash of it. The full key is stored inside the JSON for traceability.

    Example:
        _raw_path(s, "tavily", "news|soybean|2024-11", "json")
          -> WindowsPath('.../data/raw/tavily/3f2a9c1b7d4e5f60.json')
    """
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return s.path("raw", source, f"{h}.{ext}")


def cache_get_json(s: Settings, source: str, key: str) -> Any | None:
    """Return the cached JSON envelope {"_key": ..., "data": ...}, or None on a cache miss.
    Callers read the payload from ["data"].

    Example:
        cache_get_json(s, "datagov", "rid|filters|0")
          -> {"_key": "rid|filters|0", "data": {"records": [...]}}   # hit
          -> None                                                    # miss (not fetched yet)
    """
    p = _raw_path(s, source, key, "json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cache_put_json(s: Settings, source: str, key: str, obj: Any) -> None:
    """Save a JSON response. ensure_ascii=False keeps Hindi text readable in the file;
    default=str turns dates/other objects into strings instead of raising.

    Example:
        cache_put_json(s, "tavily", "news|soybean|2024-11", {"results": [...]})
          -> writes .../data/raw/tavily/<hash>.json ; returns None
    """
    p = _raw_path(s, source, key, "json")
    p.write_text(json.dumps({"_key": key, "data": obj}, ensure_ascii=False, default=str), encoding="utf-8")


def cache_get_bytes(s: Settings, source: str, key: str, ext: str = "bin") -> bytes | None:
    """Binary version of the cache (used for the USDA PSD zip download).

    Example:
        cache_get_bytes(s, "psd", "https://...psd_oilseeds_csv.zip2026-11", "zip")
          -> b'PK\\x03\\x04...'   (the zip bytes)  or  None on a miss
    """
    p = _raw_path(s, source, key, ext)
    return p.read_bytes() if p.exists() else None


def cache_put_bytes(s: Settings, source: str, key: str, data: bytes, ext: str = "bin") -> None:
    """Save raw bytes to the cache (counterpart of cache_get_bytes).

    Example:
        cache_put_bytes(s, "psd", "<url>2026-11", b'PK\\x03\\x04...', "zip")  -> writes the .zip; returns None
    """
    _raw_path(s, source, key, ext).write_bytes(data)


# =====================================================================
# Staged tables — one cleaned Parquet file per source/table.
# =====================================================================
def staged_path(s: Settings, table: str) -> Path:
    """data/staged/<table>.parquet

    Example:  staged_path(s, "mandi_daily")  ->  WindowsPath('.../data/staged/mandi_daily.parquet')
    """
    return s.path("staged", f"{table}.parquet")


def read(s: Settings, table: str) -> pd.DataFrame:
    """Load a staged table. Returns an EMPTY DataFrame if it hasn't been collected yet,
    so downstream steps can skip a missing source instead of crashing.

    Example:
        read(s, "mandi_daily")   -> DataFrame[obs_date, market, modal_price, arrivals_t, ...], 3200 rows
        read(s, "not_collected") -> empty DataFrame (0 rows, 0 columns)
    """
    p = staged_path(s, table)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def exists(s: Settings, table: str) -> bool:
    """Whether a staged table has been written yet.

    Example:  exists(s, "mandi_daily")  ->  True   |   exists(s, "psd_annual")  ->  False
    """
    return staged_path(s, table).exists()


def upsert(s: Settings, table: str, df: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    """Merge new rows into a staged table and save it.

    keys : columns that identify "the same row", e.g. ["obs_date", "market", "variety", "source"].
    If a key already exists, the NEW row replaces the old one (keep="last"), so re-running a
    collector refreshes corrected values instead of creating duplicates.
    Note: this means only the latest revision of a value is kept (no version history).
    Returns the full merged table.

    Example:
        # staged already has 1 Indore row for 2024-11-01; df has a corrected one + a new day:
        df = DataFrame([{obs_date: 2024-11-01, market: "Indore", variety: "NA", source: "datagov", modal_price: 4810},
                        {obs_date: 2024-11-02, market: "Indore", variety: "NA", source: "datagov", modal_price: 4825}])
        upsert(s, "mandi_daily", df, ["obs_date", "market", "variety", "source"])
          -> the 2024-11-01 row is overwritten, 2024-11-02 is added; returns the full merged table
    """
    keys = list(keys)
    if df is None or df.empty:
        log.info("upsert %s: nothing to write", table)
        return read(s, table)

    # Normalise the two time columns to one dtype (timezone-naive, nanosecond) in both the new and
    # the existing data. Mixed dtypes would break de-duplication and Parquet writes.
    for c in ("obs_date", "available_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    old = read(s, table)
    if not old.empty:
        for c in ("obs_date", "available_at"):
            if c in old.columns:
                old[c] = pd.to_datetime(old[c]).astype("datetime64[ns]")

    # Old rows first, new rows last -> drop_duplicates(keep="last") keeps the new version.
    out = pd.concat([old, df], ignore_index=True) if not old.empty else df.copy()
    out = out.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    out.to_parquet(staged_path(s, table), index=False)
    log.info("upsert %s: %d new rows -> %d total", table, len(df), len(out))
    return out


def write(s: Settings, table: str, df: pd.DataFrame) -> None:
    """Overwrite a staged table completely (no merge).

    Example:
        write(s, "mandi_daily", df_truncated)   # replaces the whole table with df_truncated
    """
    df.to_parquet(staged_path(s, table), index=False)
