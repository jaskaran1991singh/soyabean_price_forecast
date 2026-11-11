"""Weekly point-in-time feature table for the target soybean mandi (settings.target).

Reads the staged tables (data/staged/*.parquet) and writes the training data (data/features/).

One row per week, indexed by `week_end` (Friday, settings.project.week_ending = "W-FRI"):
  cutoff     = week_end + forecast_lag_days (Saturday): the moment the forecast is made.
  features   use ONLY rows whose available_at <= cutoff (see store.py for the two-date rule).
  targets    target_y_fwd{h} = the target mandi's weekly price h weeks later (labels, never features).

Feature groups (one function each, every column is recorded in the data dictionary):
  target      Primary Mandi price history and arrivals              (mandi_daily)
  peers       spreads/returns vs nearby mandis                (mandi_daily)
  markets     CBOT, FX, crude, palm/soy oil; Rs/qtl parity    (markets_daily)
  weather     season rain, dry spells, heat, harvest rain     (weather_daily)
  psd         global supply balance sheets                    (psd_annual)
  policy      MSP, import duty, futures ban, policy news      (seed_events.csv + agent_policy_events)
  industry    SOPA / SEA / state statistics                   (agent_industry_stats)
  products    DOC & soy-oil prices -> crush margin            (agent_product_prices)
  sentiment   relevance-weighted news sentiment               (agent_news_sentiment)
  forecasts   third-party price outlooks                      (agent_price_forecasts)
  calendar    seasonality, crop phase, Diwali                 (derived, known in advance)

Two ways a value enters a row:
  asof(...)              "latest value known at the cutoff" (levels: prices, stocks, MSP ...)
  first_usable_week(...) bucket an event into the first week whose cutoff is after it became
                         known (flows: news articles, policy events, product price quotes)
features/test_features.py::test_no_lookahead rebuilds the table from data truncated at a week's
cutoff and checks that week's features are identical — the guard against look-ahead.

This module is plain functions: build(s) assembles the week grid and calls one function per
feature group (each returns a DataFrame indexed by week_end), then joins them and adds the labels.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import store
from ..config import Settings

log = logging.getLogger(__name__)
BU_KG = 27.2155  # kg of soybeans in one bushel (CBOT quotes cents per bushel)


@dataclass
class Registry:
    """Collects one row per feature column for data_dictionary.csv: its group, source and the
    rule that decides when the value is known. A plain accumulator, passed to each feature group.

    Example:
        reg = Registry()
        reg.add(["y", "y_lag1"], "target_history", "mandi", "past weeks only")
        reg.frame()  -> DataFrame[column, group, source, availability_rule] with 2 rows
    """

    rows: list[dict] = field(default_factory=list)

    def add(self, cols, group: str, source: str, rule: str):
        """Record one or more columns under a group/source/rule.

        Example:  reg.add("y_ma4", "target_history", "mandi", "past weeks only")  # cols may be str or list
        """
        for c in ([cols] if isinstance(cols, str) else cols):
            self.rows.append({"column": c, "group": group, "source": source, "availability_rule": rule})

    def frame(self) -> pd.DataFrame:
        """The collected rows as a DataFrame (one row per column, last entry wins).

        Example:  reg.frame()  ->  DataFrame[column, group, source, availability_rule]
        """
        return pd.DataFrame(self.rows).drop_duplicates("column", keep="last")


# ======================================================================== helpers
def _lag(s: Settings) -> int:
    """Days after week_end when the forecast is issued (features may use data known by then).

    Example:  _lag(s)  ->  1   (forecast issued the Saturday after each Friday week_end)
    """
    return int(s["project"].get("forecast_lag_days", 1))


def week_of(ts: pd.Series, rule: str) -> pd.Series:
    """Map dates to the date their week ends on (e.g. Wed 2024-11-06 -> Fri 2024-11-08).

    Example:
        week_of(pd.Series(["2024-11-06", "2024-11-09"]), "W-FRI")
          -> [Timestamp('2024-11-08'), Timestamp('2024-11-15')]   # Sat rolls into the next week
    """
    return pd.to_datetime(ts).dt.to_period(rule).dt.end_time.dt.normalize()


def first_usable_week(available_at: pd.Series, rule: str, lag_days: int) -> pd.Series:
    """Earliest week whose cutoff (week_end + lag) is on/after `available_at`.

    Example with Friday weeks and lag 1: an article available Saturday 2024-11-09 counts in the
    week ending Fri 2024-11-08 (its cutoff is that Saturday); one available Sunday counts the week after.

    Example:
        first_usable_week(pd.Series(["2024-11-09", "2024-11-10"]), "W-FRI", 1)
          -> [Timestamp('2024-11-08'), Timestamp('2024-11-15')]   # Sat -> this week, Sun -> next week
    """
    return week_of(pd.to_datetime(available_at) - pd.Timedelta(days=lag_days), rule)


def asof(grid: pd.DataFrame, df: pd.DataFrame, cols: list[str], time_col: str = "available_at") -> pd.DataFrame:
    """For every week, the latest row of `df` with time_col <= that week's cutoff.

    grid     : the week table (needs a "cutoff" column)
    df       : source rows with `time_col` and the value columns `cols`
    Returns a frame aligned to grid.index (NaN where nothing was known yet).
    Uses pandas merge_asof(direction="backward"): it never looks forward in time.

    Example:
        # markets rows: (available_at=2024-10-28, value=1010), (available_at=2024-11-04, value=1020)
        # week cutoffs: 2024-11-02, 2024-11-09
        asof(grid, df, ["value"])  ->  value = [1010, 1020]   # latest known by each cutoff
    """
    if df.empty:
        return pd.DataFrame({c: np.nan for c in cols}, index=grid.index)
    d = df[[time_col] + cols].dropna(subset=[time_col]).sort_values(time_col)
    d[time_col] = d[time_col].astype("datetime64[ns]")
    # _pos remembers the original week order, because merge_asof needs both sides sorted by time.
    g = pd.DataFrame({"cutoff": grid["cutoff"].astype("datetime64[ns]").values, "_pos": np.arange(len(grid))})
    if time_col == "cutoff":                       # avoid a name clash with the grid's column
        d = d.rename(columns={"cutoff": "_tc"}); time_col = "_tc"
    m = pd.merge_asof(g.sort_values("cutoff"), d, left_on="cutoff", right_on=time_col, direction="backward")
    m = m.sort_values("_pos")
    out = m[cols].copy()
    out.index = grid.index
    return out


def _clim_z(df: pd.DataFrame, col: str, by: str = "day_of_season", season: str = "season_year") -> tuple[pd.Series, pd.Series]:
    """Departure and z-score of `col` vs the same day-of-season in PRIOR seasons only.

    E.g. cumulative rain on 15 Aug 2019 is compared with 15 Aug of 2005..2018 — never with
    later seasons, which would leak future climate into the past. shift(1) excludes the
    current season from its own baseline.

    Example:
        # rain_season_cum on day_of_season=75: 400mm this year, prior years averaged 300mm (std 50)
        _clim_z(d, "rain_season_cum")  ->  (dep=+100mm, z=+2.0)   # a wet year vs. normal
    """
    d = df[[season, by, col]].copy().sort_values([by, season])
    grp = d.groupby(by)[col]
    mean = grp.transform(lambda x: x.expanding().mean().shift(1))
    std = grp.transform(lambda x: x.expanding().std().shift(1))
    dep = (d[col] - mean).reindex(df.index)
    z = ((d[col] - mean) / std.replace(0, np.nan)).reindex(df.index)
    return dep, z


def week_grid(s: Settings) -> pd.DataFrame:
    """Every week_end (Friday) from start_date to end_date, with its cutoff (Saturday).

    Example:
        week_grid(s).head(2)
          ->   week_end     cutoff
             2015-01-02  2015-01-03
             2015-01-09  2015-01-10      (index = week_end)
    """
    rule = s.week_rule
    w = pd.date_range(pd.Timestamp(s.start_date), pd.Timestamp(s.end_date), freq=rule)
    grid = pd.DataFrame({"week_end": w})
    grid["cutoff"] = grid["week_end"] + pd.Timedelta(days=_lag(s))
    grid.index = grid["week_end"]
    return grid


# ================================================================= target & price history
def target(s: Settings, grid: pd.DataFrame, reg: Registry, mandi: pd.DataFrame) -> pd.DataFrame:
    """Primary Mandi's weekly price `y`, its history features, and arrivals.

    Daily: one modal price per day (arrival-weighted across varieties when arrivals exist).
    Weekly: y = mean of the week's daily modal prices; y_close = last trading day's price.

    Example:
        target(s, grid, reg, mandi_daily)  ->  DataFrame indexed by week_end with columns
          y, y_close, y_lag1..4, y_ma4, y_ma13, ret_1w/4w/13w, vol_8w, y_yoy, arrivals_wk, ...
          e.g. week 2024-11-08: y=4950.0, y_lag1=4900.0, ret_1w=0.0102, arrivals_wk=18500.0
    """
    rule = s.week_rule
    t = s["target"]
    m = mandi[mandi["market"].str.lower() == t["market"].lower()].copy()
    if t.get("varieties"):
        m = m[m["variety"].isin(t["varieties"])]
    if m.empty:
        raise RuntimeError("No mandi rows for the target market; run `collect mandi` first.")
    # Several sources can report the same day: keep the Agmarknet file row (it has arrivals).
    m["_pri"] = (m["source"] != "agmarknet_csv").astype(int)
    m = m.sort_values("_pri").drop_duplicates(["obs_date", "variety"])

    def day_agg(g):
        w = g["arrivals_t"].fillna(0)
        modal = np.average(g["modal_price"], weights=w) if w.sum() > 0 else g["modal_price"].mean()
        return pd.Series({"modal": modal, "min": g["min_price"].min(), "max": g["max_price"].max(),
                          "arrivals": g["arrivals_t"].sum(min_count=1)})

    daily = m.groupby("obs_date")[["modal_price", "min_price", "max_price", "arrivals_t"]].apply(day_agg)
    daily["week_end"] = week_of(daily.index.to_series(), rule).values
    wk = daily.groupby("week_end").agg(y=("modal", "mean"), y_close=("modal", "last"), y_min=("min", "min"),
                                       y_max=("max", "max"), arrivals_wk=("arrivals", lambda x: x.sum(min_count=1)),
                                       trading_days=("modal", "size"))
    # Compute history over the FULL span of mandi data (data before start_date feeds the
    # first weeks' lags / 52-week changes), then trim to the grid at the end.
    full = pd.date_range(min(wk.index.min(), grid.index.min()), grid.index.max(), freq=rule)
    out = wk.reindex(full)
    out = out[out.index <= grid.index.max()]
    h = out["y"]
    f = pd.DataFrame(index=full)
    f["y"] = h                                         # this week's price (known at cutoff)
    f["y_close"] = out["y_close"]
    f["y_range_pct"] = (out["y_max"] - out["y_min"]) / h   # intra-week price dispersion
    f["trading_days"] = out["trading_days"].fillna(0)  # 0 = closed week (holiday/strike)
    hf = h.ffill(limit=3)  # carry the price over up to 3 closed weeks, for lag features only
    for k in (1, 2, 3, 4):
        f[f"y_lag{k}"] = hf.shift(k)
    f["y_ma4"] = hf.rolling(4, min_periods=2).mean()
    f["y_ma13"] = hf.rolling(13, min_periods=6).mean()
    lr = np.log(hf).diff()                             # weekly log return
    f["ret_1w"] = lr
    f["ret_4w"] = np.log(hf / hf.shift(4))
    f["ret_13w"] = np.log(hf / hf.shift(13))
    f["vol_8w"] = lr.rolling(8, min_periods=4).std()   # recent volatility
    f["y_vs_ma13"] = hf / f["y_ma13"] - 1              # distance from the quarterly trend
    f["y_yoy"] = np.log(hf / hf.shift(52))             # change vs same week last year
    reg.add(["y", "y_close", "y_range_pct", "trading_days"], "target_history", "mandi", "week's own trades (known at cutoff)")
    reg.add([c for c in f.columns if c.startswith(("y_lag", "y_ma", "ret_", "vol_", "y_vs", "y_yoy"))],
            "target_history", "mandi", "past weeks only")
    # Arrivals (tonnes): supply pressure. Crop year for the season total starts 1 October.
    a = out["arrivals_wk"]
    f["arrivals_wk"] = a
    f["arrivals_ma4"] = a.rolling(4, min_periods=2).mean()
    f["arrivals_yoy"] = np.log((a + 1) / (a.shift(52) + 1))
    crop_year = np.where(full.month >= 10, full.year, full.year - 1)
    # Season-to-date total; stays NaN until the first week with arrivals data in that crop year.
    f["arrivals_season_cum"] = a.fillna(0).groupby(crop_year).cumsum().where(a.notna().groupby(crop_year).cummax())
    reg.add(["arrivals_wk", "arrivals_ma4", "arrivals_yoy", "arrivals_season_cum"], "arrivals", "mandi", "week's own arrivals")
    return f.reindex(grid.index)


def peers(s: Settings, grid: pd.DataFrame, reg: Registry, mandi: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """For each peer mandi: weekly price spread vs the target mandi and its weekly return.
    Peers often move first (e.g. Latur reacts to Maharashtra arrivals), so these can lead the target.

    Example:
        peers(s, grid, reg, mandi_daily, y)  ->  DataFrame indexed by week_end with columns
          peer_ujjain_spread, peer_ujjain_ret_1w, ..., peer_avg_spread
          e.g. week 2024-11-08: peer_ujjain_spread=-30.0 (Ujjain Rs 30/qtl below Indore)
    """
    rule = s.week_rule
    f = pd.DataFrame(index=grid.index)
    spreads = []
    for p in s.get("peer_markets", []):
        m = mandi[mandi["market"].str.lower() == p["market"].lower()]
        if m.empty:
            continue
        wk = m.assign(week_end=week_of(m["obs_date"], rule)).groupby("week_end")["modal_price"].mean()
        wk = wk.reindex(grid.index)
        name = re.sub(r"\W+", "_", p["market"].lower())
        f[f"peer_{name}_spread"] = wk - y
        f[f"peer_{name}_ret_1w"] = np.log(wk.ffill(limit=2)).diff()
        spreads.append(f[f"peer_{name}_spread"])
    if spreads:
        f["peer_avg_spread"] = pd.concat(spreads, axis=1).mean(axis=1)
    reg.add(list(f.columns), "peer_mandis", "mandi", "same week, obs_date <= week_end")
    return f


# ================================================================= global markets
def markets(s: Settings, grid: pd.DataFrame, reg: Registry, mk: pd.DataFrame) -> pd.DataFrame:
    """Latest known close of each market series, plus derived parity / crush / spread features.

    Example:
        markets(s, grid, reg, markets_daily)  ->  DataFrame indexed by week_end with columns
          mkt_cbot_soybean, mkt_usdinr, ..., cbot_soy_rs_qtl, cbot_board_crush_usd_bu, mkt_*_ret_1w
          e.g. week 2024-11-08: mkt_cbot_soybean=1012.5, cbot_soy_rs_qtl=3790.0
    """
    f = pd.DataFrame(index=grid.index)
    if mk.empty:
        return f
    for series, g in mk.groupby("series"):
        f[f"mkt_{series}"] = asof(grid, g, ["value"])["value"].values

    def col(n):
        return f.get(f"mkt_{n}")

    # CBOT soybeans in Rs/quintal: cents/bu -> USD/bu -> USD/kg -> USD/qtl -> Rs/qtl.
    # Comparable with the target mandi price (international parity, before freight/duty).
    if col("cbot_soybean") is not None and col("usdinr") is not None:
        f["cbot_soy_rs_qtl"] = col("cbot_soybean") / 100 / BU_KG * 100 * col("usdinr")
    # CBOT board crush (USD/bu): value of 44 lb meal (0.022 short ton) + 11 lb oil - 1 bu beans.
    # High crush margin -> crushers bid up beans.
    if all(col(n) is not None for n in ("cbot_soybean", "cbot_soymeal", "cbot_soyoil")):
        f["cbot_board_crush_usd_bu"] = col("cbot_soymeal") * 0.022 + col("cbot_soyoil") / 100 * 11 - col("cbot_soybean") / 100
    # Soy oil premium over palm oil: wide premium -> India imports palm instead, softer soy oil demand.
    if col("palm_oil_usd_t") is not None and col("soyoil_usd_t") is not None:
        f["soy_palm_oil_spread_usd_t"] = col("soyoil_usd_t") - col("palm_oil_usd_t")
    for n in ("cbot_soybean", "cbot_soymeal", "cbot_soyoil", "usdinr", "usdbrl", "brent"):
        c = col(n)
        if c is not None:
            f[f"mkt_{n}_ret_1w"] = np.log(c).diff()
            f[f"mkt_{n}_ret_4w"] = np.log(c / c.shift(4))
    reg.add(list(f.columns), "global_markets", "yfinance/fred", "latest close with available_at <= cutoff (FRED +35d)")
    return f


# ================================================================= weather / season quality
def weather(s: Settings, grid: pd.DataFrame, reg: Registry, wd: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date weather over the catchment (average of the configured points).

    Season year runs from 1 June (sowing) — so Jan-May weeks carry the finished previous
    season's values, which is the crop being marketed then.
      rain_season_cum      rain since 1 Jun (counted until 31 Oct)
      rain_season_dep/z    vs prior seasons on the same day (deficit -> lower yield)
      rain_harvest_cum/z   rain in the harvest window (moisture, damage -> quality discounts)
      dry_spell_max/z      longest run of dry days in the growing period
      heat_days_flowering  days above heat_threshold_c during flowering
      season_quality_risk  composite: mean of harvest-rain, dry-spell and heat z-scores
                           + half of any rain deficit z. Higher = more yield/quality risk.
      rain_wk / tmax_wk / tmin_wk   this week's values

    Example:
        weather(s, grid, reg, weather_daily)  ->  DataFrame indexed by week_end with columns
          wx_rain_season_cum, wx_rain_season_z, wx_dry_spell_max, wx_season_quality_risk, wx_rain_wk, ...
          e.g. week 2024-08-30: wx_rain_season_cum=720.0, wx_rain_season_z=-0.8 (drier than normal)
    """
    rule = s.week_rule
    f = pd.DataFrame(index=grid.index)
    if wd.empty:
        return f
    cfg = s["weather"]["season"]
    d = wd.groupby("obs_date")[["precip_mm", "tmax", "tmin", "tmean"]].mean().sort_index()
    d = d.asfreq("D")                                  # one row per calendar day (gaps -> NaN)
    idx = d.index
    d["season_year"] = np.where(idx.month >= 6, idx.year, idx.year - 1)
    sy = d["season_year"].astype(str).values
    jun1 = pd.DatetimeIndex(pd.to_datetime(pd.Series(sy) + "-06-01").values)
    d["day_of_season"] = (idx - jun1).days

    def window_mask(a, b):
        """True on days between MM-DD `a` and MM-DD `b` of each row's season year."""
        lo = pd.DatetimeIndex(pd.to_datetime(pd.Series(sy) + "-" + a).values)
        hi = pd.DatetimeIndex(pd.to_datetime(pd.Series(sy) + "-" + b).values)
        return pd.Series((idx >= lo) & (idx <= hi), index=idx)

    p = d["precip_mm"].fillna(0)
    in_season = window_mask("06-01", "10-31")
    # cumsum within each season; outside the window adds 0, so the total "holds" afterwards.
    d["rain_season_cum"] = (p * in_season).groupby(d["season_year"]).cumsum()
    d["rain_harvest_cum"] = (p * window_mask(*cfg["harvest_window"])).groupby(d["season_year"]).cumsum()
    # Dry-spell length: running count of consecutive dry days (reset on every wet day).
    dry = (p < cfg["dry_day_mm"]) & window_mask(cfg["sowing_start"], "09-30")
    run = dry.groupby((~dry).cumsum()).cumsum()
    d["dry_spell_max"] = run.groupby(d["season_year"]).cummax()
    heat = (d["tmax"] > cfg["heat_threshold_c"]) & window_mask(*cfg["flowering"])
    d["heat_days_flowering"] = heat.astype(int).groupby(d["season_year"]).cumsum()
    d["rain_days_season"] = ((p >= cfg["dry_day_mm"]) & in_season).astype(int).groupby(d["season_year"]).cumsum()
    dep, z = _clim_z(d, "rain_season_cum")
    d["rain_season_dep_mm"], d["rain_season_z"] = dep, z
    _, d["rain_harvest_z"] = _clim_z(d, "rain_harvest_cum")
    _, d["dry_spell_z"] = _clim_z(d, "dry_spell_max")
    _, d["heat_days_z"] = _clim_z(d, "heat_days_flowering")
    d["season_quality_risk"] = d[["rain_harvest_z", "dry_spell_z", "heat_days_z"]].clip(-3, 3).mean(axis=1) \
        + 0.5 * (-d["rain_season_z"]).clip(lower=0).fillna(0)
    d["available_at"] = idx + pd.Timedelta(days=1)     # a day's weather is known the next day
    cols = ["rain_season_cum", "rain_season_dep_mm", "rain_season_z", "rain_harvest_cum", "rain_harvest_z",
            "dry_spell_max", "dry_spell_z", "heat_days_flowering", "heat_days_z", "rain_days_season", "season_quality_risk"]
    snap = asof(grid, d.reset_index(drop=True), cols)
    for c in cols:
        f[f"wx_{c}"] = snap[c].values
    wk = d.assign(week_end=week_of(d.index.to_series(), rule).values).groupby("week_end").agg(
        rain_wk=("precip_mm", "sum"), tmax_wk=("tmax", "mean"), tmin_wk=("tmin", "mean"))
    wk = wk.reindex(grid.index)
    for c in wk.columns:
        f[f"wx_{c}"] = wk[c]
    reg.add(list(f.columns), "weather_season", "open-meteo ERA5", "daily obs <= week_end; climatology from prior seasons only")
    return f


# ================================================================= USDA PSD
def psd(s: Settings, grid: pd.DataFrame, reg: Registry, psd_df: pd.DataFrame) -> pd.DataFrame:
    """Latest marketing year's balance-sheet values known at each cutoff (15-month lag, see usda_psd.py).
    Column names: psd_<commodity>_<country>_<attribute>, e.g. psd_soybean_brazil_production.

    Example:
        psd(s, grid, reg, psd_annual)  ->  DataFrame indexed by week_end with columns
          psd_soybean_world_production, psd_soybean_brazil_ending_stocks, psd_soybean_world_stocks_to_use, ...
          e.g. week 2024-11-08: psd_soybean_world_stocks_to_use=0.27
    """
    f = pd.DataFrame(index=grid.index)
    if psd_df.empty:
        return f
    for (cty, com, att), g in psd_df.groupby(["country", "commodity", "attribute"]):
        name = "psd_" + re.sub(r"\W+", "_", f"{com}_{cty}_{att}".lower())
        f[name] = asof(grid, g, ["value"])["value"].values
    # Stocks-to-use: the standard tightness measure (low = tight supply = supportive prices).
    es, dc = f.get("psd_soybean_world_ending_stocks"), f.get("psd_soybean_world_domestic_consumption")
    if es is not None and dc is not None:
        f["psd_soybean_world_stocks_to_use"] = es / dc
    reg.add(list(f.columns), "global_supply", "USDA PSD", "MY value usable from MY start + 15 months (latest-revision data)")
    return f


# ================================================================= policy (seed + agent)
def policy(s: Settings, grid: pd.DataFrame, reg: Registry, events: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Policy levels in force each week, and counts of recent bullish/bearish decisions.

    An event counts for week W only if it was KNOWN by W's cutoff (known_at = latest of
    announcement and availability) and IN FORCE on W (effective_from <= W <= effective_to).

    Example:
        policy(s, grid, reg, events, y)  ->  DataFrame indexed by week_end with columns
          msp_effective, spread_over_msp, duty_crude_soyoil_pct, futures_suspended, policy_bull_events_4w, ...
          e.g. week 2024-11-08: msp_effective=4892.0, spread_over_msp=58.0, futures_suspended=1
    """
    rule, lag = s.week_rule, _lag(s)
    f = pd.DataFrame(index=grid.index)
    if events.empty:
        return f
    ev = events.copy()
    for c in ("announced_at", "effective_from", "effective_to"):
        ev[c] = pd.to_datetime(ev[c], errors="coerce")
    ev["known_at"] = ev[["announced_at", "available_at"]].max(axis=1) if "available_at" in ev else ev["announced_at"]
    ev["known_at"] = ev["known_at"].fillna(ev["effective_from"])

    def step(etype: str) -> pd.Series:
        """Weekly level of a numeric policy (MSP, duty %): the latest-known event in force."""
        e = ev[(ev["event_type"] == etype) & ev["value"].notna()]
        vals = []
        for W, cut in zip(grid["week_end"], grid["cutoff"]):
            cand = e[(e["known_at"] <= cut) & (e["effective_from"] <= W) & (e["effective_to"].isna() | (e["effective_to"] >= W))]
            vals.append(cand.sort_values("known_at")["value"].iloc[-1] if len(cand) else np.nan)
        return pd.Series(vals, index=grid.index)

    # MSP: the floor currently in force, and the next season's MSP once announced (May/June),
    # which the market starts pricing before it takes effect in October.
    f["msp_effective"] = step("msp")
    msp = ev[ev["event_type"] == "msp"].sort_values("known_at")
    f["msp_latest_announced"] = asof(grid, msp.rename(columns={"known_at": "_k"}), ["value"], "_k")["value"].values
    f["msp_change_pct"] = f["msp_latest_announced"] / f["msp_effective"] - 1
    f["spread_over_msp"] = y - f["msp_effective"]      # < 0: mandi below MSP -> procurement/support likely
    f["ratio_to_msp"] = y / f["msp_effective"]
    # Crude soy oil effective import duty: lower duty -> cheaper imported oil -> weaker crush margin.
    f["duty_crude_soyoil_pct"] = step("import_duty_crude_soyoil")
    chg = f["duty_crude_soyoil_pct"].diff().fillna(0)
    f["duty_crude_soyoil_chg_8w"] = chg.rolling(8, min_periods=1).sum()
    last = pd.Series(np.where(chg != 0, np.arange(len(chg)), np.nan), index=chg.index).ffill()
    f["weeks_since_duty_change"] = np.arange(len(chg)) - last
    # 1 while SEBI's suspension of soybean futures is in force (and known).
    fs = ev[ev["event_type"] == "futures_suspension"]
    f["futures_suspended"] = 0
    for _, r in fs.iterrows():
        on = (grid["week_end"] >= r["effective_from"]) & (grid["cutoff"] >= r["known_at"])
        if pd.notna(r["effective_to"]):
            on &= grid["week_end"] <= r["effective_to"]
        f.loc[on.values, "futures_suspended"] = 1
    # Agent-reported decisions tagged bullish/bearish for soybean, counted over 4 weeks.
    if "direction" not in ev:
        ev["direction"] = None
    ev["week"] = first_usable_week(ev["known_at"], rule, lag)
    for d, nm in (("bullish", "policy_bull_events_4w"), ("bearish", "policy_bear_events_4w")):
        cnt = ev[ev["direction"] == d].groupby("week").size().reindex(grid.index, fill_value=0)
        f[nm] = cnt.rolling(4, min_periods=1).sum()
    reg.add(list(f.columns), "policy", "seed_events.csv + agent policy_events", "announcement/known date <= cutoff")
    return f


# ================================================================= agent: industry statistics
def _to_lakh(value: float, unit: str) -> float:
    """Convert a printed unit to lakh (1 lakh = 100,000) tonnes or hectares; NaN if unknown.
    Checks run from most to least specific, e.g. 'million tonnes' before 'tonnes'.

    Example:
        _to_lakh(12.5, "million tonnes")  -> 125.0     # 1 million t = 10 lakh t
        _to_lakh(125, "lakh tonnes")      -> 125.0
        _to_lakh(1250000, "tonnes")       -> 12.5
        _to_lakh(5, "bushels")            -> nan       # unrecognised unit
    """
    u = (unit or "").lower()
    if "million" in u or "mn" in u.split():
        return value * 10
    if "thousand" in u or "'000" in u or "000 t" in u:
        return value / 100
    if "lakh" in u or "lac" in u:
        return value
    if "crore" in u:
        return value * 100
    if "tonne" in u or u.strip() in ("t", "tons", "mt"):   # "MT" = metric tonnes in Indian usage
        return value / 1e5
    if "hectare" in u or "ha" in u.split():
        return value / 1e5
    return np.nan


def industry(s: Settings, grid: pd.DataFrame, reg: Registry, st: pd.DataFrame) -> pd.DataFrame:
    """Latest published value of each statistic and how old it is (weeks since publication).

    Example:
        industry(s, grid, reg, agent_industry_stats)  ->  DataFrame indexed by week_end with columns
          ind_sopa_crop_estimate_lt, ind_sopa_crop_estimate_lt_age_w, ind_sea_soyoil_import_lt, ...
          e.g. week 2024-11-08: ind_sopa_crop_estimate_lt=125.0, ..._age_w=3.0 (published 3 weeks ago)
    """
    f = pd.DataFrame(index=grid.index)
    if st.empty:
        return f
    st = st.copy()
    st["value_lakh"] = [_to_lakh(v, u) for v, u in zip(st["value"], st["unit"])]
    st = st.dropna(subset=["value_lakh"])
    for metric, g in st.groupby("metric"):
        if metric == "other":
            continue
        snap = asof(grid, g.assign(_t=g["available_at"]), ["value_lakh", "_t"])
        f[f"ind_{metric}"] = snap["value_lakh"].values
        # Age lets a model discount stale figures (e.g. last season's crop estimate in August).
        f[f"ind_{metric}_age_w"] = ((grid["cutoff"] - pd.to_datetime(snap["_t"])).dt.days / 7).values
    reg.add(list(f.columns), "industry_stats", "agent: SOPA/SEA/state agri", "published date <= cutoff")
    return f


# ================================================================= agent: product prices -> crush margin
def products(s: Settings, grid: pd.DataFrame, reg: Registry, pp: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Weekly soy product prices in Rs/qtl and the domestic crush margin.

    crush_gpm = meal_yield x DOC price + oil_yield x oil price   (value of 1 qtl of seed's products)
    crush_margin = crush_gpm - target mandi seed price - processing cost
    A wide margin means crushers can pay more for seed. Quotes are carried forward up to 4 weeks.

    Example:
        products(s, grid, reg, agent_product_prices, y)  ->  DataFrame indexed by week_end with columns
          px_soymeal_doc_local_rs_qtl, px_soy_oil_refined_local_rs_qtl, crush_gpm_rs_qtl, crush_margin_rs_qtl, ...
          e.g. week 2024-11-08: crush_gpm_rs_qtl=5200.0, crush_margin_rs_qtl=-100.0 (margin squeezed)
    """
    rule, lag = s.week_rule, _lag(s)
    f = pd.DataFrame(index=grid.index)
    if pp.empty:
        return f
    pp = pp.copy()
    conv = {"rs_per_tonne": 0.1, "rs_per_10kg": 10.0, "rs_per_qtl": 1.0}   # USD quotes are skipped
    pp["rs_qtl"] = pp["price"] * pp["unit"].map(conv)
    pp = pp.dropna(subset=["rs_qtl"])
    pp["week"] = first_usable_week(pp[["available_at", "obs_date"]].max(axis=1), rule, lag)
    for prod, g in pp.groupby("product"):
        # median across reports in a week resists a single mis-extracted price
        wk = g.groupby("week")["rs_qtl"].median().reindex(grid.index).ffill(limit=4)
        f[f"px_{prod}_rs_qtl"] = wk
    econ = s["economics"]
    meal, oil = f.get("px_soymeal_doc_local_rs_qtl"), f.get("px_soy_oil_refined_local_rs_qtl")
    if meal is not None and oil is not None:
        gpm = econ["meal_yield"] * meal + econ["oil_yield"] * oil
        f["crush_gpm_rs_qtl"] = gpm
        f["crush_margin_rs_qtl"] = gpm - y - econ["processing_cost_rs_qtl"]
    if "px_soy_plant_delivery_local_rs_qtl" in f:
        f["plant_vs_mandi_spread"] = f["px_soy_plant_delivery_local_rs_qtl"] - y
    reg.add(list(f.columns), "soy_products", "agent: news price reports", "report date <= cutoff, ffill <= 4w")
    return f


# ================================================================= agent: sentiment & forecasts
def sentiment(s: Settings, grid: pd.DataFrame, reg: Registry, ns: pd.DataFrame) -> pd.DataFrame:
    """News sentiment weighted by relevance x confidence, per week, over 4 weeks, as an EWM,
    and per topic. NaN (not 0) when there were no articles, so "no news" != "neutral news".

    Example:
        sentiment(s, grid, reg, agent_news_sentiment)  ->  DataFrame indexed by week_end with columns
          news_count_wk, news_sent_wk, news_sent_4w, news_sent_ewm, news_sent_supply_4w, ...
          e.g. week 2024-11-08: news_count_wk=4, news_sent_4w=+0.35 (mildly bullish), no-news week = NaN
    """
    rule, lag = s.week_rule, _lag(s)
    f = pd.DataFrame(index=grid.index)
    if ns.empty:
        return f
    ns = ns.copy()
    ns["week"] = first_usable_week(ns["available_at"], rule, lag)
    ns["w"] = ns["relevance"].clip(0, 1) * ns["confidence"].clip(0, 1)
    ns["ws"] = ns["w"] * ns["sentiment"]
    g = ns.groupby("week")
    wk = pd.DataFrame({"ws": g["ws"].sum(), "w": g["w"].sum(), "n": g.size()}).reindex(grid.index, fill_value=0)
    f["news_count_wk"] = wk["n"]
    f["news_sent_wk"] = (wk["ws"] / wk["w"].replace(0, np.nan))
    r4 = wk.rolling(4, min_periods=1).sum()
    f["news_sent_4w"] = r4["ws"] / r4["w"].replace(0, np.nan)
    f["news_count_4w"] = r4["n"]
    f["news_sent_ewm"] = (wk["ws"].ewm(halflife=3).mean() / wk["w"].ewm(halflife=3).mean().replace(0, np.nan))
    if "topics" in ns:
        for topic in ("supply", "demand", "policy", "weather", "global"):
            sub = ns[ns["topics"].fillna("").str.contains(topic)]   # topics stored as "supply|policy"
            sg = sub.groupby("week")
            t = pd.DataFrame({"ws": sg["ws"].sum(), "w": sg["w"].sum()}).reindex(grid.index, fill_value=0).rolling(4, min_periods=1).sum()
            f[f"news_sent_{topic}_4w"] = t["ws"] / t["w"].replace(0, np.nan)
    reg.add(list(f.columns), "news_sentiment", "agent: Tavily news + LLM scoring", "article publish date <= cutoff")
    return f


def forecasts(s: Settings, grid: pd.DataFrame, reg: Registry, fc: pd.DataFrame) -> pd.DataFrame:
    """Third-party outlooks published in the last 4 weeks: average direction (+1 up / -1 down),
    how many there were, and the average expected % change where one was given.

    Example:
        forecasts(s, grid, reg, agent_price_forecasts)  ->  DataFrame indexed by week_end with columns
          fc3p_direction_4w, fc3p_count_4w, fc3p_exp_change_pct_4w
          e.g. week 2024-11-08: fc3p_direction_4w=+0.5 (net bullish), fc3p_count_4w=2
    """
    rule, lag = s.week_rule, _lag(s)
    f = pd.DataFrame(index=grid.index)
    if fc.empty:
        return f
    fc = fc.copy()
    fc["week"] = first_usable_week(fc["available_at"], rule, lag)
    fc["dir"] = fc["direction"].map({"up": 1, "down": -1, "sideways": 0})
    g = fc.groupby("week")
    wk = pd.DataFrame({"dir_sum": g["dir"].sum(), "n": g.size(),
                       "chg_sum": g["expected_change_pct"].sum(min_count=1), "chg_n": g["expected_change_pct"].count()}
                      ).reindex(grid.index).fillna(0)
    r = wk.rolling(4, min_periods=1).sum()
    f["fc3p_direction_4w"] = r["dir_sum"] / r["n"].replace(0, np.nan)
    f["fc3p_count_4w"] = r["n"]
    f["fc3p_exp_change_pct_4w"] = r["chg_sum"] / r["chg_n"].replace(0, np.nan)
    reg.add(list(f.columns), "third_party_forecasts", "agent: forecasts in news/reports", "publish date <= cutoff")
    return f


# ================================================================= calendar
def calendar(s: Settings, grid: pd.DataFrame, reg: Registry) -> pd.DataFrame:
    """Seasonality known in advance: week-of-year (as sin/cos so week 52 sits next to week 1),
    month, crop phase dummies, weeks since the 1 Oct crop year start, and Diwali timing
    (mandis close for several days around Diwali, which falls in peak harvest arrivals).

    Example:
        calendar(s, grid, reg)  ->  DataFrame indexed by week_end with columns
          woy_sin, woy_cos, month, phase_sowing..phase_lean, weeks_since_oct1, diwali_week, weeks_to_diwali
          e.g. week 2024-11-08: month=11, phase_peak_arrivals=1, weeks_to_diwali=0
    """
    rule = s.week_rule
    idx = grid.index
    f = pd.DataFrame(index=idx)
    woy = idx.isocalendar().week.astype(int).values
    f["woy_sin"], f["woy_cos"] = np.sin(2 * np.pi * woy / 52.18), np.cos(2 * np.pi * woy / 52.18)
    f["month"] = idx.month
    m = idx.month
    phase = np.select([np.isin(m, [6, 7]), m == 8, m == 9, np.isin(m, [10, 11, 12]), np.isin(m, [1, 2, 3])],
                      ["sowing", "growing", "early_harvest", "peak_arrivals", "post_harvest"], "lean")
    for p in ("sowing", "growing", "early_harvest", "peak_arrivals", "post_harvest", "lean"):
        f[f"phase_{p}"] = (phase == p).astype(int)
    oct1 = pd.to_datetime(np.where(m >= 10, idx.year, idx.year - 1).astype(str) + "-10-01")
    f["weeks_since_oct1"] = ((idx - oct1).days // 7).values
    try:
        import holidays
        hd = holidays.India(years=range(idx.year.min(), idx.year.max() + 2))
        diwali = pd.to_datetime([d for d, n in hd.items() if "iwali" in n])
        dw = week_of(pd.Series(diwali), rule)
        f["diwali_week"] = idx.isin(dw).astype(int)
        f["weeks_to_diwali"] = [min([(d - w).days // 7 for d in dw if d >= w], default=99) for w in idx]
    except Exception as e:  # noqa: BLE001 - calendar still useful without Diwali
        log.warning("holidays unavailable: %s", e)
    reg.add(list(f.columns), "calendar", "derived", "known in advance")
    return f


def _events(s: Settings) -> pd.DataFrame:
    """Policy events = your seed_events.csv + agent policy_events (confidence >= 0.7).
    Where both have the same event type in the same month, the seed row wins.

    Example:
        _events(s)  ->  DataFrame[event_type, announced_at, effective_from, effective_to, value,
                                  direction, available_at, origin]
          e.g. row (event_type="msp", effective_from=2024-10-01, value=4892, origin="seed")
    """
    seed_p = s.root / "config" / "seed_events.csv"
    seed = pd.read_csv(seed_p) if seed_p.exists() else pd.DataFrame()
    if not seed.empty:
        seed["available_at"] = pd.to_datetime(seed["announced_at"])
        seed["origin"] = "seed"
    ag = store.read(s, "agent_policy_events")
    if not ag.empty:
        ag = ag[ag["confidence"] >= 0.7].copy()
        ag["origin"] = "agent"
        if not seed.empty:
            sk = set(zip(seed["event_type"], pd.to_datetime(seed["effective_from"]).dt.to_period("M")))
            ek = list(zip(ag["event_type"], pd.to_datetime(ag["effective_from"], errors="coerce").dt.to_period("M")))
            ag = ag[[k not in sk for k in ek]]
    frames = [d for d in (seed, ag) if not d.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["available_at"] = pd.to_datetime(out["available_at"])
    return out


# ================================================================= assemble
def build(s: Settings) -> tuple[pd.DataFrame, Registry]:
    """Run every feature group, add the forward targets, and join on week_end.
    Returns the full weekly table and the column registry (for the data dictionary).

    Example:
        X, reg = build(s)
        X.shape           -> (618, 180)        # weeks x (cutoff + features + target_* labels)
        X.index.name      -> 'week_end'
        [c for c in X.columns if c.startswith("target_")]  -> ['target_y_fwd1', 'target_logret_fwd1', ...]
    """
    grid = week_grid(s)
    reg = Registry()
    mandi = store.read(s, "mandi_daily")
    base = target(s, grid, reg, mandi)
    y = base["y"]
    events = _events(s)
    parts = [base, peers(s, grid, reg, mandi, y), markets(s, grid, reg, store.read(s, "markets_daily")),
             weather(s, grid, reg, store.read(s, "weather_daily")), psd(s, grid, reg, store.read(s, "psd_annual")),
             policy(s, grid, reg, events, y), industry(s, grid, reg, store.read(s, "agent_industry_stats")),
             products(s, grid, reg, store.read(s, "agent_product_prices"), y),
             sentiment(s, grid, reg, store.read(s, "agent_news_sentiment")),
             forecasts(s, grid, reg, store.read(s, "agent_price_forecasts")),
             calendar(s, grid, reg)]
    # Targets: price h weeks ahead and the log return to it. These are LABELS — never use them as
    # model inputs (every target_* column is excluded from the Time-LLM file).
    tg = {}
    for h in s["target"]["horizons_weeks"]:
        tg[f"target_y_fwd{h}"] = y.shift(-h)
        tg[f"target_logret_fwd{h}"] = np.log(y.shift(-h) / y)
        reg.add([f"target_y_fwd{h}", f"target_logret_fwd{h}"], "target", "mandi", f"FUTURE: week t+{h} (label, not a feature)")
    X = pd.concat([grid[["cutoff"]], *parts, pd.DataFrame(tg)], axis=1)
    X.index.name = "week_end"
    return X, reg


# ======================================================================== outputs
def write_outputs(s: Settings, X: pd.DataFrame, reg: Registry) -> dict:
    """Write the training files to data/features/:

    training_weekly.parquet / .csv   every feature + every target (for tree / linear models)
    timellm_<market>_<commodity>.csv Time-LLM / TimesFM layout: date first, numeric covariates,
                                     target `y` last, no target_* labels. Keeps columns >= 60%
                                     filled, forward-fills, then drops rows with remaining gaps.
                                     Named from settings (s.target_slug), e.g. timellm_indore_soyabean.csv.
    data_dictionary.csv              column, group, source, availability rule, % filled, first date
    coverage_report.md               % filled per feature group per year

    Example:
        write_outputs(s, X, reg)
          -> writes the 5 files into data/features/ and returns
             {"rows": 618, "features": 160, "timellm_cols": 120, "dir": ".../data/features"}
    """
    out_dir = s.path("features", "x").parent
    X.to_parquet(out_dir / "training_weekly.parquet")
    X.to_csv(out_dir / "training_weekly.csv")
    feats = [c for c in X.columns if not c.startswith("target_") and c not in ("cutoff",) and pd.api.types.is_numeric_dtype(X[c])]
    covs = [c for c in feats if c != "y"]
    tl = X[covs + ["y"]].copy()
    tl = tl[tl["y"].notna()]
    tl = tl.loc[:, tl.notna().mean() >= 0.6].ffill().dropna()
    tl.index = tl.index.strftime("%Y-%m-%d")
    tl.index.name = "date"
    tl.to_csv(out_dir / f"timellm_{s.target_slug}.csv")
    d = reg.frame()
    nn = X.notna().mean()
    d["non_null_pct"] = d["column"].map(lambda c: round(100 * nn.get(c, np.nan), 1))
    d["first_valid"] = d["column"].map(lambda c: X[c].first_valid_index().date() if c in X and X[c].notna().any() else None)
    d.to_csv(out_dir / "data_dictionary.csv", index=False)
    _coverage_report(X, d, out_dir / "coverage_report.md")
    return {"rows": len(X), "features": len(feats), "timellm_cols": tl.shape[1], "dir": str(out_dir)}


def _coverage_report(X: pd.DataFrame, d: pd.DataFrame, path) -> None:
    """Markdown table: for each feature group, the average % of filled values per year.
    Use it to decide which groups have enough history to train on.

    Example:
        _coverage_report(X, d, path)  ->  writes a markdown table like
          | group          | 2015 | 2016 | ... |
          | target_history |  100 |  100 | ... |
          | news_sentiment |    0 |   12 | ... |
    """
    lines = ["# Feature coverage", "", f"Weeks: {len(X)} ({X.index.min().date()} → {X.index.max().date()})",
             f"Target weeks with price: {int(X['y'].notna().sum())}", "", "## By group (mean non-null % by year)", ""]
    years = sorted(X.index.year.unique())
    lines.append("| group | " + " | ".join(map(str, years)) + " |")
    lines.append("|---" * (len(years) + 1) + "|")
    for grp, cols in d.groupby("group")["column"]:
        cols = [c for c in cols if c in X]
        if not cols:
            continue
        cov = X[cols].notna().mean(axis=1).groupby(X.index.year).mean()
        lines.append(f"| {grp} | " + " | ".join(f"{100 * cov.get(y, 0):.0f}" for y in years) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_all(s: Settings) -> dict:
    """Entry point used by `python -m soyfc build`.

    Example:
        build_all(s)  ->  {"rows": 618, "features": 160, "timellm_cols": 120, "dir": ".../data/features"}
                          (and writes training_weekly.*, timellm_*.csv, data_dictionary.csv, coverage_report.md)
    """
    X, reg = build(s)
    return write_outputs(s, X, reg)
