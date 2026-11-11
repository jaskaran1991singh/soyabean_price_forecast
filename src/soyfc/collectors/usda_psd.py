"""USDA FAS PSD (Production, Supply & Distribution) — global soybean balance sheets.

Downloads the public bulk oilseeds CSV (one zip, no key) and keeps, for soybean / soymeal /
soy oil, the countries and attributes listed in settings.usda_psd (production, ending stocks,
exports, imports, domestic consumption). A "World" total is computed by summing countries.

Look-ahead caution: PSD only holds the LATEST revision of every number. The 2024/25 Brazil
crop you download today already includes revisions made long after that season was being
traded. To avoid training on knowledge the market didn't have yet, a marketing year's value
is only made available `availability_lag_months` (15) after the marketing year starts.
The trade-off: the current season's global supply comes from the research agent
(news / WASDE coverage), not from this table.

Output table `psd_annual`: obs_date (marketing-year anchor, 1 Sep), market_year, country,
commodity, attribute, value, unit (usually "(1000 MT)"), source, available_at
"""
from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd

from .. import store
from ..config import Settings
from .http import get

log = logging.getLogger(__name__)


def collect(s: Settings) -> pd.DataFrame:
    """Download the USDA PSD oilseeds zip and stage soybean/meal/oil balance sheets.

    Example:
        collect(s) -> DataFrame[obs_date, market_year, country, commodity, attribute, value, unit,
                                source="usda_psd", available_at]
                      e.g. (2023-09-01, 2023, "Brazil", "soybean", "Production", 160000.0, "(1000 MT)",
                            available_at=2024-12-01)  # MY2023 usable only 15 months after MY start
                      and writes data/staged/psd_annual.parquet
    """
    cfg = s["usda_psd"]
    # Cache the zip per calendar month: re-running within a month doesn't re-download ~MBs.
    month_key = cfg["url"] + pd.Timestamp.today().strftime("%Y-%m")
    blob = store.cache_get_bytes(s, "psd", month_key, "zip")
    if blob is None:
        blob = get(cfg["url"], timeout=180).content
        store.cache_put_bytes(s, "psd", month_key, blob, "zip")

    # The zip contains a single CSV with every oilseed commodity and country.
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        df = pd.read_csv(z.open(name), dtype={"Commodity_Code": str})

    # Keep only our commodities. Codes are 7 digits; leading zeros can be lost (813100 -> 0813100).
    codes = {v: k for k, v in cfg["commodities"].items()}          # "2222000" -> "soybean"
    df["Commodity_Code"] = df["Commodity_Code"].str.zfill(7)
    df = df[df["Commodity_Code"].isin(codes) & df["Attribute_Description"].isin(cfg["attributes"])]

    # PSD has no ready-made World row in this file: sum all countries per commodity/year/attribute.
    world = df.groupby(["Commodity_Code", "Market_Year", "Attribute_Description", "Unit_Description"],
                       as_index=False)["Value"].sum().assign(Country_Name="World")
    df = pd.concat([df, world], ignore_index=True)
    df = df[df["Country_Name"].isin(cfg["countries"])]

    # Each country's marketing year starts in a different month (Brazil Feb, US Sep, India Oct);
    # PSD's own "Month" column is the last-update month, not the MY start. A common 1-Sep anchor
    # plus the 15-month lag keeps this simple and conservative.
    df["obs_date"] = pd.to_datetime(df["Market_Year"].astype(int).astype(str) + "-09-01")
    df["available_at"] = df["obs_date"] + pd.DateOffset(months=cfg.get("availability_lag_months", 15))

    out = pd.DataFrame({
        "obs_date": df["obs_date"], "market_year": df["Market_Year"].astype(int),
        "country": df["Country_Name"], "commodity": df["Commodity_Code"].map(codes),
        "attribute": df["Attribute_Description"], "value": df["Value"].astype(float),
        "unit": df["Unit_Description"], "source": "usda_psd", "available_at": df["available_at"],
    })
    # A few years before start_date so the first training weeks already have a "latest known" year.
    out = out[out["market_year"] >= pd.Timestamp(s.start_date).year - 3]
    return store.upsert(s, "psd_annual", out, ["market_year", "country", "commodity", "attribute"])