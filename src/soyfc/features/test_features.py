"""Feature-builder tests — no network, synthetic mandi data only.

The important one is the point-in-time guard (test_no_lookahead): a week's features must not
change when data that only became available AFTER that week's cutoff is removed. If they do,
a feature is leaking the future into the training data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from soyfc import store  # noqa: E402
from soyfc.config import load_settings  # noqa: E402
from soyfc.features.build import build, build_all  # noqa: E402


def _settings(tmp_path, start="2019-01-01", end="2022-12-31"):
    s = load_settings()
    s.raw["project"]["data_dir"] = str(tmp_path)
    s.raw["project"]["start_date"] = start
    s.raw["project"]["end_date"] = end
    return s


def _stage_mandi(s, start="2017-06-01", end="2022-12-31"):
    """Stage a synthetic daily price series for the configured target market (enough history
    for 52-week features). Pulls the market/state from settings, so the test isn't tied to Indore."""
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(0)
    price = 4500 + np.cumsum(rng.normal(0, 25, len(days)))
    t = s["target"]
    df = pd.DataFrame({
        "obs_date": days, "state": t["state"], "district": t["district"], "market": t["market"],
        "commodity": t["commodity"], "variety": "NA", "min_price": price - 50, "max_price": price + 50,
        "modal_price": price, "arrivals_t": rng.uniform(100, 500, len(days)),
        "source": "datagov", "available_at": days + pd.Timedelta(days=1)})
    store.upsert(s, "mandi_daily", df, ["obs_date", "market", "variety", "source"])


def test_build_produces_training_files(tmp_path):
    s = _settings(tmp_path)
    _stage_mandi(s)
    res = build_all(s)
    assert res["rows"] > 100 and res["features"] > 10
    out = s.path("features", "x").parent
    # timellm file is named from the configured market (s.target_slug), not a fixed "indore".
    for name in ("training_weekly.parquet", f"timellm_{s.target_slug}.csv", "data_dictionary.csv"):
        assert (out / name).exists()
    X = pd.read_parquet(out / "training_weekly.parquet")
    assert "y" in X and any(c.startswith("target_y_fwd") for c in X.columns)   # labels present


def test_no_lookahead(tmp_path):
    """Removing data available only after week W's cutoff must not change W's features."""
    s = _settings(tmp_path)
    _stage_mandi(s)

    X_full, _ = build(s)
    # A week in the middle, where 52-week history exists and forward targets are not the point.
    W = X_full.index[120]
    cutoff = X_full.loc[W, "cutoff"]

    # Rebuild from mandi data truncated at W's cutoff (drop anything only knowable later).
    mandi = store.read(s, "mandi_daily")
    store.write(s, "mandi_daily", mandi[mandi["available_at"] <= cutoff])
    X_trunc, _ = build(s)

    # Compare every feature column (targets look forward by design, so they are excluded).
    feats = [c for c in X_full.columns if not c.startswith("target_") and c != "cutoff"]
    a, b = X_full.loc[W, feats], X_trunc.loc[W, feats]
    mismatch = [c for c in feats if not ((pd.isna(a[c]) and pd.isna(b[c])) or a[c] == b[c])]
    assert not mismatch, f"look-ahead in columns: {mismatch}"
