"""Daily weather for the target market's soybean catchment (settings.weather.points),
from Open-Meteo (free, no API key).

Two Open-Meteo endpoints are combined:
  ARCHIVE   ERA5 reanalysis — consistent history back decades, but ~5 days behind real time
  FORECAST  forecast model with `past_days` — covers the most recent days ERA5 hasn't reached
Where both cover a day, ERA5 is kept.

Output table `weather_daily` (one row per day per point in settings.weather.points):
  obs_date, point, precip_mm, tmax, tmin, tmean (deg C), et0 (reference evapotranspiration, mm),
  source ("era5" / "om_recent"), available_at (= obs_date + 1 day)
The season features (cumulative rain, dry spells, harvest rain ...) are derived later in
features/build.py; this module only stores raw daily values.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .. import store
from ..config import Settings
from .http import get

log = logging.getLogger(__name__)

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
# Daily variables requested from both endpoints.
DAILY = "precipitation_sum,temperature_2m_max,temperature_2m_min,temperature_2m_mean,et0_fao_evapotranspiration"
# Open-Meteo names -> our column names.
RENAME = {"time": "obs_date", "precipitation_sum": "precip_mm", "temperature_2m_max": "tmax",
          "temperature_2m_min": "tmin", "temperature_2m_mean": "tmean", "et0_fao_evapotranspiration": "et0"}


def _fetch(s: Settings, url: str, pt: dict, start: str, end: str | None, past_days: int | None = None) -> pd.DataFrame:
    """Download daily weather for one point.

    pt        : {"name", "lat", "lon"} from settings
    start/end : date range (archive endpoint)
    past_days : if given, ask the forecast endpoint for the last N days instead of a date range
    Days are in Indian time (Asia/Kolkata), so a "day" matches the mandi trading day.

    Example:
        _fetch(s, ARCHIVE, {"name": "Indore", "lat": 22.72, "lon": 75.86}, "2024-06-01", "2024-06-30")
          -> DataFrame[obs_date, precip_mm, tmax, tmin, tmean, et0, point="Indore"], 30 rows
    """
    params = {"latitude": pt["lat"], "longitude": pt["lon"], "daily": DAILY, "timezone": "Asia/Kolkata"}
    if past_days:
        params.update({"past_days": past_days, "forecast_days": 1})
    else:
        params.update({"start_date": start, "end_date": end})
    key = f"{url}|{sorted(params.items())}"
    # Recent-days requests are never cached: they change daily. History is cached (below).
    cached = None if past_days else store.cache_get_json(s, "openmeteo", key)
    data = cached["data"] if cached else get(url, params=params).json()
    # Only cache periods that ended >10 days ago, i.e. ERA5 values that won't change any more.
    if not cached and not past_days and end and end < (date.today() - timedelta(days=10)).isoformat():
        store.cache_put_json(s, "openmeteo", key, data)
    df = pd.DataFrame(data["daily"]).rename(columns=RENAME)
    df["point"] = pt["name"]
    return df


def collect(s: Settings) -> pd.DataFrame:
    """Collect all points from climatology_start (e.g. 2005) to end_date and stage them.

    History starts before the training window (2015) so that "rain vs normal" features have
    earlier seasons to compare against from the very first training year.

    Example:
        collect(s) -> DataFrame[obs_date, point, precip_mm, tmax, tmin, tmean, et0, source, available_at]
                      (e.g. 7 points x ~7000 days) and writes data/staged/weather_daily.parquet
    """
    pts = s["weather"]["points"]
    start = min(pd.Timestamp(s.start_date), pd.Timestamp(s["weather"].get("climatology_start", s.start_date)))
    end = pd.Timestamp(s.end_date)
    # ERA5 lags ~5 days; ask the archive only up to 6 days ago.
    archive_end = min(end, pd.Timestamp(date.today()) - pd.Timedelta(days=6))
    frames = []
    for pt in pts:
        # One request per calendar year: small responses, and each closed year is cached once.
        for y in range(start.year, archive_end.year + 1):
            a = max(start, pd.Timestamp(f"{y}-01-01"))
            b = min(archive_end, pd.Timestamp(f"{y}-12-31"))
            if a > b:
                continue
            frames.append(_fetch(s, ARCHIVE, pt, a.date().isoformat(), b.date().isoformat()).assign(source="era5"))
        # Fill the last few days that ERA5 doesn't cover yet from the forecast model.
        if end > archive_end:
            frames.append(_fetch(s, FORECAST, pt, "", None, past_days=30).assign(source="om_recent"))
        log.info("weather %s done", pt["name"])
    df = pd.concat(frames, ignore_index=True)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    # Drop today and the forecast day: only completed days are observations.
    df = df[df["obs_date"] <= pd.Timestamp(date.today()) - pd.Timedelta(days=1)]
    # Where ERA5 and the forecast model overlap, keep ERA5 (rank 0 sorts first, kept by drop_duplicates).
    df["_rank"] = (df["source"] != "era5").astype(int)
    df = df.sort_values("_rank").drop_duplicates(["obs_date", "point"]).drop(columns="_rank")
    df["available_at"] = df["obs_date"] + pd.Timedelta(days=1)   # a day's weather is known next day
    return store.upsert(s, "weather_daily", df, ["obs_date", "point"])