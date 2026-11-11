"""Mandi (APMC) daily prices & arrivals for the target market and the peer markets
(both from settings: target + peer_markets; Indore is just the default target).

Three sources, tried in the order given by settings.mandi_sources.order:
  mandi_api  free keyless REST wrapper over data.gov.in mandi prices (mandi-api.onrender.com).
             Rate-limited to 100 calls / 15 min. Prices; arrivals only if the API returns them.
  datagov    data.gov.in Agmarknet API (needs DATA_GOV_IN_API_KEY). Gives prices, not arrivals.
  csv        Agmarknet "Price & Arrival" report files that you download yourself and drop into
             data/manual/agmarknet/. The only bulk route that also carries arrivals (tonnes).

All are cleaned into one common table, `mandi_daily`:
  obs_date      trading day
  state, district, market, commodity, variety
  min_price, max_price, modal_price   Rs per quintal (modal = most common traded price)
  arrivals_t    arrivals in tonnes (empty when the source doesn't report them)
  source        "mandi_api", "datagov" or "agmarknet_csv"
  available_at  obs_date + 1 day (Agmarknet posts the day's prices the same evening / next day)
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

from .. import store
from ..config import Settings, env
from .http import get

log = logging.getLogger(__name__)

# Column order of the staged table, and the columns that identify one row.
COLS = ["obs_date", "state", "district", "market", "commodity", "variety",
        "min_price", "max_price", "modal_price", "arrivals_t", "source", "available_at"]
KEYS = ["obs_date", "market", "variety", "source"]

# Header names differ between the data.gov.in API ("Modal_Price", "modal_x0020_price") and
# the Agmarknet report exports ("Modal Price (Rs./Quintal)"). Each target column lists the
# header spellings we accept (after lower-casing and stripping the "(Rs./Quintal)" suffix).
_ALIASES = {
    "obs_date": ["arrival_date", "arrivaldate", "price date", "price_date", "pricedate", "reported date", "date"],
    "state": ["state", "state name", "statename"],
    "district": ["district", "district name", "districtname"],
    "market": ["market", "market name", "marketname", "mandi_name", "apmc"],
    "commodity": ["commodity", "crop"],
    "variety": ["variety"],
    "min_price": ["min_price", "min price", "minprice", "min_x0020_price", "minimum price", "min"],
    "max_price": ["max_price", "max price", "maxprice", "max_x0020_price", "maximum price", "max"],
    # avg_modal_price: /prices/history returns averaged modal prices per date
    "modal_price": ["modal_price", "modal price", "modalprice", "modal_x0020_price", "avg_modal_price",
                    "average_modal_price", "modal"],
    "arrivals_t": ["arrivals", "arrivals (tonnes)", "arrival", "arrivals_tonnes", "quantity"],
}


def _norm(s: str) -> str:
    """Normalise a header for matching: lower-case, drop '(Rs./Quintal)', collapse spaces.

    Example:  _norm("Modal Price (Rs./Quintal)")  ->  'modal price'
    """
    s = str(s).strip().lower()
    s = re.sub(r"\(rs\.?/?\s*quintal\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whatever headers a file/API uses to our standard column names.

    Matching per target column, first hit wins:
      1. exact match with an alias, or
      2. header starts with an alias longer than 4 characters (e.g. "arrivals (tonnes) total").
    A source column is only ever mapped once. obs_date is matched first, so "arrival_date"
    becomes the date and is not mistaken for "arrivals".

    Example:
        columns ["Arrival_Date", "Market", "Modal_Price", "Arrivals"]
          -> renamed to ["obs_date", "market", "modal_price", "arrivals_t"]
    """
    mapping = {}
    normed = {c: _norm(c) for c in df.columns}
    for target, aliases in _ALIASES.items():
        for c, n in normed.items():
            if c in mapping:            # this source column is already used
                continue
            if n in aliases or any(n.startswith(a) for a in aliases if len(a) > 4):
                mapping[c] = target
                break
    return df.rename(columns=mapping)


def _fix_unit_slips(df: pd.DataFrame) -> pd.DataFrame:
    """Repair the common Agmarknet data-entry slip of a missing or extra zero.

    For each market, the median of plausible prices is the reference. A price that is far off
    but lands within 0.6x-1.6x of that median after multiplying or dividing by 10 is corrected
    (e.g. 495 -> 4950 or 49500 -> 4950). Anything else is left alone; out-of-range values are
    dropped later by the price bounds.

    Example:
        Indore modal_price [4900, 495, 4950, 49500]  (median ~4950)
          -> [4900, 4950, 4950, 4950]   (495 x10, 49500 /10; 4900 already in range)
    """
    for col in ("min_price", "max_price", "modal_price"):
        for mkt, idx in df.groupby("market").groups.items():
            v = df.loc[idx, col]
            med = v[(v > 500) & (v < 50000)].median()
            if pd.isna(med):
                continue
            ok = v.between(0.6 * med, 1.6 * med)
            up = ~ok & (v * 10).between(0.6 * med, 1.6 * med)
            down = ~ok & (v / 10).between(0.6 * med, 1.6 * med)
            df.loc[up[up].index, col] = v[up] * 10
            df.loc[down[down].index, col] = v[down] / 10
    return df


def _parse_dates(col: pd.Series) -> pd.Series:
    """Parse a column that may mix "2026-10-28" (ISO) and "28/10/2026" / "28 Oct 2026" (day-first).

    ISO strings are parsed with an explicit format first, because day-first parsing can swap
    month and day in "2026-10-08". Everything else is parsed day-first, row by row ("mixed").

    Example:
        ["2024-11-01", "28/10/2026", "28 Oct 2026", "bad"]
          -> [Timestamp('2024-11-01'), Timestamp('2026-10-28'), Timestamp('2026-10-28'), NaT]
    """
    txt = col.astype("string").str.strip()
    iso = txt.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")
    out[iso] = pd.to_datetime(txt[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    out[~iso] = pd.to_datetime(txt[~iso], format="mixed", dayfirst=True, errors="coerce")
    return out


def clean(df: pd.DataFrame, s: Settings, source: str) -> pd.DataFrame:
    """Turn raw rows from any source into the standard `mandi_daily` layout.

    Example:
        raw row {"Arrival_Date": "01/11/2024", "Market": "Indore APMC", "Modal_Price": "4,950", "Variety": None}
          -> cleaned row {obs_date: 2024-11-01, market: "Indore", modal_price: 4950.0, variety: "NA",
                          source: "datagov", available_at: 2024-11-02}   (out-of-range prices dropped)
    """
    df = normalize_columns(df.copy())
    for c in COLS:                       # make sure every expected column exists
        if c not in df.columns:
            df[c] = pd.NA
    # Dates come as "01/11/2024", "01 Nov 2024" or "2024-11-01", sometimes mixed in one batch.
    df["obs_date"] = _parse_dates(df["obs_date"])
    # Numbers may be strings with thousands separators ("4,950") or placeholders ("NR", "-").
    for c in ("min_price", "max_price", "modal_price", "arrivals_t"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").replace({"NR": None, "-": None}), errors="coerce")
    for c in ("state", "district", "market", "commodity", "variety"):
        df[c] = df[c].astype("string").str.strip()
    # "Indore APMC" / "Indore(F&V)" -> "Indore", so the same mandi always has one name.
    df["market"] = df["market"].str.replace(r"\s*\(.*\)$", "", regex=True).str.replace(r"\s*APMC$", "", regex=True)
    df = _fix_unit_slips(df)
    # Drop rows still outside the plausible price range or with an unreadable date.
    lo, hi = s["target"]["price_floor"], s["target"]["price_ceiling"]
    df = df[df["modal_price"].between(lo, hi) & df["obs_date"].notna()].copy()
    df["source"] = source
    df["available_at"] = df["obs_date"] + pd.Timedelta(days=1)   # point-in-time key (see store.py)
    df["variety"] = df["variety"].fillna("NA")                    # key column must not be empty
    return df[COLS]


def _markets(s: Settings) -> list[dict]:
    """Target market first, then the peer markets from settings.

    Example:
        _markets(s) -> [{"state": "Madhya Pradesh", "district": "Indore", "market": "Indore"},
                        {"state": "Madhya Pradesh", "district": "Ujjain", "market": "Ujjain"}, ...]
    """
    t = s["target"]
    return [{"state": t["state"], "district": t["district"], "market": t["market"]}] + list(s.get("peer_markets", []))


# ============================================================================ mandi-api (keyless)
_last_call = [0.0]   # time of the previous mandi-api request (module-level, shared by all calls)


def _mandi_api_get(s: Settings, path: str, params: dict) -> object:
    """GET one mandi-api endpoint, respecting the rate limit, with a same-day cache.

    path   : e.g. "/markets", "/prices/history"
    params : query parameters (requests URL-encodes spaces in "Madhya Pradesh")
    Responses are cached per calendar day: re-running the same day costs no API calls,
    and the next day fetches fresh prices.

    Example:
        _mandi_api_get(s, "/markets", {"state": "Madhya Pradesh"})
          -> {"success": True, "data": ["Indore", "Ujjain", "Dewas", ...]}   (shape varies by endpoint)
    """
    cfg = s["mandi_sources"]["mandi_api"]
    key = f"{path}|{sorted(params.items())}|{date.today().isoformat()}"
    cached = store.cache_get_json(s, "mandi_api", key)
    if cached:
        return cached["data"]
    # Space calls out so we never exceed 100 requests per 15 minutes.
    wait = cfg["min_seconds_between_calls"] - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    r = get(cfg["base_url"] + path, params=params, timeout=cfg.get("timeout_seconds", 90),
            headers={"Accept": "application/json"})
    _last_call[0] = time.time()
    data = r.json()
    if isinstance(data, dict) and data.get("success") is False:   # API's own error envelope
        raise RuntimeError(f"mandi-api {path} error: {data.get('error')}")
    store.cache_put_json(s, "mandi_api", key, data)
    return data


def _records(payload) -> list[dict]:
    """Find the list of row-dicts inside a response, whatever it's wrapped in.

    The docs don't publish the success schema, so accept a bare list or a list under common
    keys ("data", "records", "prices", ...), searching one nested level down.

    Example:
        _records({"data": [{"date": "2024-11-01", "modal": 4950}]})  -> [{"date": "2024-11-01", "modal": 4950}]
        _records({"success": True})                                 -> []
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for k in ("data", "records", "prices", "results", "history", "items"):
            v = payload.get(k)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
            if isinstance(v, dict):
                inner = _records(v)
                if inner:
                    return inner
    return []


def _names(payload) -> list[str]:
    """List of names from /states or /markets: plain strings, or dicts with a name-like field.

    Accepts a bare list, or the first list found under any key of a dict response
    (e.g. {"success": true, "data": [...]} or {"states": [...]}).

    Example:
        _names({"data": ["Indore", "Ujjain"]})                    -> ["Indore", "Ujjain"]
        _names({"data": [{"market_name": "Indore APMC"}]})        -> ["Indore APMC"]
    """
    items = payload if isinstance(payload, list) else []
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                items = v
                break
            if isinstance(v, dict):          # one nested level, e.g. {"data": {"markets": [...]}}
                inner = _names(v)
                if inner:
                    return inner
    out = []
    for r in items:
        if isinstance(r, str):
            out.append(r)
        elif isinstance(r, dict):
            for k in ("name", "market", "state", "market_name", "state_name"):
                if r.get(k):
                    out.append(str(r[k]))
                    break
    return out


def _resolve_market(s: Settings, state: str, market: str) -> str | None:
    """Map our market name ("Indore") to the API's exact spelling ("Indore APMC", "Indore(F&V)"...).

    Picks the API market whose cleaned name equals ours; returns None if the state or market
    isn't covered (the API supports only a handful of states).

    Example:
        _resolve_market(s, "Madhya Pradesh", "Indore")  -> "Indore APMC"   (the API's spelling)
        _resolve_market(s, "Kerala", "Indore")          -> None            (state not covered)
    """
    names = _names(_mandi_api_get(s, "/markets", {"state": state}))
    want = market.strip().lower()
    for n in names:
        base = re.sub(r"\s*\(.*\)$", "", n).replace("APMC", "").strip().lower()
        if base == want:
            return n
    return None


def probe_mandi_api(s: Settings) -> dict:
    """Show what the API covers and what its rows look like, before a full pull.
    Run with: python -m soyfc probe-mandi-api

    Example:
        probe_mandi_api(s) -> {"states": ["Madhya Pradesh", ...], "resolved_market": "Indore APMC",
                               "/prices/history": {"n_rows": 420, "sample_rows": [...],
                                                   "date_range": ["2015-01-02", "2024-11-01"]}}
    """
    t = s["target"]
    out = {"states": _names(_mandi_api_get(s, "/states", {}))}
    if t["state"] not in out["states"]:
        out["warning"] = f"{t['state']} not in supported states — this source can't serve {t['market']}."
        return out
    markets = _names(_mandi_api_get(s, "/markets", {"state": t["state"]}))
    out["markets_matching_target"] = [m for m in markets if t["market"].lower() in m.lower()]
    mk = _resolve_market(s, t["state"], t["market"])
    out["resolved_market"] = mk
    if mk:
        for path in ("/prices", "/prices/history"):
            payload = _mandi_api_get(s, path, {"state": t["state"], "commodity": t["commodity"], "market": mk})
            rows = _records(payload)
            out[path] = {"top_level_keys": list(payload)[:10] if isinstance(payload, dict) else "list",
                         "n_rows": len(rows), "sample_rows": rows[:2],
                         "date_range": [min((r.get("arrival_date") or r.get("date") or "") for r in rows),
                                        max((r.get("arrival_date") or r.get("date") or "") for r in rows)] if rows else None}
    return out


def collect_mandi_api(s: Settings) -> pd.DataFrame:
    """Pull /prices/history and /prices (latest day) for every configured market.

    `market` is always passed: without it, /prices/history returns a state-wide AVERAGE,
    which is not what we want as the target mandi.

    Example:
        collect_mandi_api(s) -> DataFrame in `mandi_daily` layout for the target + peer mandis
                                [obs_date, market, modal_price, arrivals_t, source="mandi_api", ...]
    """
    commodity = s["target"]["commodity"]
    states = set(_names(_mandi_api_get(s, "/states", {})))
    frames = []
    for m in _markets(s):
        if m["state"] not in states:
            log.info("mandi-api: state %s not supported, skipping %s", m["state"], m["market"])
            continue
        mk = _resolve_market(s, m["state"], m["market"])
        if not mk:
            log.warning("mandi-api: market %s not found in %s", m["market"], m["state"])
            continue
        for path in ("/prices/history", "/prices"):
            try:
                rows = _records(_mandi_api_get(s, path, {"state": m["state"], "commodity": commodity, "market": mk}))
            except Exception as e:  # noqa: BLE001 - one market failing shouldn't stop the rest
                log.warning("mandi-api %s %s failed: %s", path, m["market"], e)
                continue
            if rows:
                # Standardise headers per response BEFORE combining: /prices and /prices/history
                # may name the same field differently ("date" vs "arrivalDate").
                df = normalize_columns(pd.DataFrame(rows))
                # Fill identity columns the endpoint may omit (history rows are often just date+price).
                for col, val in (("state", m["state"]), ("district", m["district"]), ("market", mk), ("commodity", commodity)):
                    if col not in df.columns:
                        df[col] = val
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = clean(pd.concat(frames, ignore_index=True), s, "mandi_api")
    return out[out["obs_date"] >= s.start_date]


# ============================================================================ data.gov.in
DATAGOV = "https://api.data.gov.in/resource/{rid}"


def probe_datagov(s: Settings) -> dict:
    """Fetch 3 rows from each configured resource and show its title, size, field names and a
    sample row, so you can confirm the resource IDs/field names before a full pull.
    Run with: python -m soyfc probe-datagov

    Example:
        probe_datagov(s) -> {"historical_resource_id": {"title": "Variety-wise Daily Market Prices",
                                                        "total": 1200000, "fields": ["state", "market", ...],
                                                        "sample": [{...}]}, "current_resource_id": {...}}
    """
    key = env("DATA_GOV_IN_API_KEY", required=True)
    out = {}
    for name in ("historical_resource_id", "current_resource_id"):
        rid = s["mandi_sources"]["datagov"][name]
        r = get(DATAGOV.format(rid=rid), params={"api-key": key, "format": "json", "limit": 3}).json()
        out[name] = {"title": r.get("title"), "total": r.get("total"),
                     "fields": [f.get("id") for f in r.get("field", [])], "sample": r.get("records", [])[:1]}
    return out


def _datagov_fetch(s: Settings, rid: str, filters: dict) -> pd.DataFrame:
    """Download all pages of one data.gov.in resource for the given filters.

    The API returns at most `page_size` rows per call, so we keep increasing `offset` until a
    short page comes back. Each page is cached, so an interrupted run resumes cheaply.

    Example:
        _datagov_fetch(s, "35985678-...", {"State": "Madhya Pradesh", "Market": "Indore", "Commodity": "Soyabean"})
          -> DataFrame of raw API rows [{"state": ..., "market": "Indore", "modal_price": "4950", ...}, ...]
    """
    key = env("DATA_GOV_IN_API_KEY", required=True)
    size = s["mandi_sources"]["datagov"]["page_size"]
    rows, offset = [], 0
    while True:
        params = {"api-key": key, "format": "json", "limit": size, "offset": offset}
        params.update({f"filters[{k}]": v for k, v in filters.items()})
        ck = f"{rid}|{sorted(filters.items())}|{offset}"   # cache key: resource + filters + page (no API key)
        cached = store.cache_get_json(s, "datagov", ck)
        data = cached["data"] if cached else get(DATAGOV.format(rid=rid), params=params).json()
        if not cached:
            store.cache_put_json(s, "datagov", ck, data)
        recs = data.get("records", [])
        rows.extend(recs)
        offset += size
        if len(recs) < size or offset >= int(data.get("total", 0) or 0):   # last page reached
            break
    return pd.DataFrame(rows)


def collect_datagov(s: Settings) -> pd.DataFrame:
    """Pull every configured market from both data.gov.in resources (history + current).

    Example:
        collect_datagov(s) -> DataFrame in `mandi_daily` layout, source="datagov"
                              (empty DataFrame with the standard columns if nothing came back)
    """
    cfg = s["mandi_sources"]["datagov"]
    commodity = s["target"]["commodity"]
    frames = []
    for m in _markets(s):
        for rid in (cfg["historical_resource_id"], cfg["current_resource_id"]):
            # The two resources name their filter fields differently ("State" vs "state.keyword").
            # Try the first spelling; if it returns nothing or errors, try the second.
            for f in ({"State": m["state"], "Market": m["market"], "Commodity": commodity},
                      {"state.keyword": m["state"], "market": m["market"], "commodity": commodity}):
                try:
                    df = _datagov_fetch(s, rid, f)
                except Exception as e:  # noqa: BLE001 - log and try the next variant
                    log.warning("datagov %s %s failed: %s", rid[:8], m["market"], e)
                    continue
                if not df.empty:
                    frames.append(normalize_columns(df))   # the two resources use different headers
                    break
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = clean(pd.concat(frames, ignore_index=True), s, "datagov")
    return out[out["obs_date"] >= s.start_date]


# ============================================================================ manual Agmarknet files
def collect_csv(s: Settings) -> pd.DataFrame:
    """Read every Agmarknet export in data/manual/agmarknet/ (CSV, XLS, XLSX or HTML).

    Example:
        # data/manual/agmarknet/ holds Indore_2023.xls, Indore_2024.csv
        collect_csv(s) -> DataFrame in `mandi_daily` layout, source="agmarknet_csv" (these carry arrivals)
    """
    d = s.root / s["mandi_sources"]["csv"]["import_dir"]
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in sorted(d.glob("*")):
        if p.suffix.lower() not in (".csv", ".xls", ".xlsx", ".html", ".htm"):
            continue                      # ignore .gitkeep, notes, etc.
        try:
            frames.append(normalize_columns(_read_any(p)))   # files may differ in headers
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the rest
            log.warning("could not read %s: %s", p.name, e)
    if not frames:
        log.info("no manual Agmarknet files in %s", d)
        return pd.DataFrame(columns=COLS)
    df = clean(pd.concat(frames, ignore_index=True), s, "agmarknet_csv")
    # Some exports only name the commodity in the report title, so keep blanks plus soy rows.
    df = df[df["commodity"].isna() | df["commodity"].str.contains("soya", case=False, na=True)]
    return df


def _read_any(p: Path) -> pd.DataFrame:
    """Open one export file whatever its real format.

    Example:  _read_any(Path("Indore_2024.xls"))  ->  DataFrame of the file's rows
              (an ".xls" that is really an HTML table is read with pd.read_html)
    """
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in (".html", ".htm"):
        return pd.read_html(p)[0]
    try:
        return pd.read_excel(p)
    except Exception:  # Agmarknet's ".xls" downloads are often an HTML table renamed to .xls
        return pd.read_html(p)[0]


# ============================================================================ entry point
def collect(s: Settings) -> pd.DataFrame:
    """Run the configured sources and merge them into data/staged/mandi_daily.parquet.

    All sources are kept side by side (source is part of the key); the feature builder
    prefers the CSV row when several have the same day, because it carries arrivals.

    Example:
        collect(s) -> merged `mandi_daily` DataFrame and writes data/staged/mandi_daily.parquet
                      (returns the already-staged table unchanged if every source came back empty)
    """
    frames = []
    for src in s["mandi_sources"]["order"]:
        fn = {"mandi_api": collect_mandi_api, "datagov": collect_datagov, "csv": collect_csv}[src]
        if src == "datagov" and not env("DATA_GOV_IN_API_KEY"):
            log.info("skipping datagov (no DATA_GOV_IN_API_KEY)")
            continue
        try:
            df = fn(s)
        except Exception as e:  # noqa: BLE001 - e.g. API asleep/down: fall through to other sources
            log.warning("mandi source %s failed: %s", src, e)
            continue
        log.info("mandi source %s: %d rows", src, len(df))
        frames.append(df)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return store.read(s, "mandi_daily")   # nothing new: return what's already staged
    return store.upsert(s, "mandi_daily", pd.concat(frames, ignore_index=True), KEYS)