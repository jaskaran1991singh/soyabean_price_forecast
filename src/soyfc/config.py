"""Settings and secrets loader.

Every module in the pipeline gets its configuration from here:
  * load_settings()  -> reads config/settings.yaml (+ loads .env into environment variables)
  * Settings         -> thin wrapper around the YAML dict with a few convenience properties
  * env()            -> the ONLY way code reads API keys (never from settings.yaml)
"""
from __future__ import annotations  # allows "str | None" style type hints on older Python

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

# python-dotenv is optional: if it isn't installed, keys must already be set as
# real environment variables (e.g. in Windows "Environment Variables" dialog).
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Project root folder = two levels above this file:
#   <root>/src/soyfc/config.py  ->  parents[0]=soyfc, parents[1]=src, parents[2]=<root>
# Resolving it from the file location (not the current working directory) means the
# pipeline finds config/ and data/ no matter which folder you launch it from.
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    """Holds the parsed settings.yaml.

    raw  : the full YAML content as a nested dict
    root : project root folder (used to resolve relative paths)

    Example:
        s = load_settings()
        s["target"]["market"]   -> 'Indore'          # dict-style access into the YAML
        s.start_date            -> '2015-01-01'       # convenience property
    """

    raw: dict[str, Any]
    root: Path

    # Allow dict-style access: s["target"]["market"]
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    # Dict-style .get with a default, for optional sections: s.get("peer_markets", [])
    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def start_date(self) -> str:
        """First date of history to collect, e.g. '2015-01-01'.

        Example:  s.start_date  ->  '2015-01-01'
        """
        return self.raw["project"]["start_date"]

    @property
    def end_date(self) -> str:
        """Last date to collect. `end_date: null` in YAML means 'today'.

        Example:  s.end_date  ->  '2026-11-03'   (when end_date is null in the YAML)
        """
        return self.raw["project"].get("end_date") or date.today().isoformat()

    @property
    def week_rule(self) -> str:
        """pandas weekly frequency string. 'W-FRI' = weeks ending on Friday.

        Example:  s.week_rule  ->  'W-FRI'
        """
        return self.raw["project"].get("week_ending", "W-FRI")

    @property
    def data_dir(self) -> Path:
        """Folder where raw/, staged/ and features/ live.
        Relative paths in settings are taken relative to the project root.

        Example:  s.data_dir  ->  WindowsPath('C:/Users/Admin/soyabean_price_forecast/data')
        """
        p = Path(self.raw["project"].get("data_dir", "data"))
        return p if p.is_absolute() else self.root / p

    def market_terms(self) -> dict[str, str]:
        """The target market/commodity names used in agent prompts, search hints and messages.

        Everything that names the market in user-facing text (the researcher prompt, the Tavily
        search hints, the probe warning) pulls from here, so changing `target.market` in
        settings.yaml is enough to retarget the pipeline — nothing says "Indore" in code.
        Optional Hindi aliases (`target.market_hi` / `target.state_hi`) let the Hindi search
        hints localise too; when they are absent they fall back to the English names.

        Example:
            s.market_terms()  ->  {"mandi": "Indore", "state": "Madhya Pradesh",
                                   "commodity": "Soyabean", "mandi_hi": "इंदौर",
                                   "state_hi": "मध्य प्रदेश"}
        """
        t = self.raw["target"]
        return {
            "mandi": t["market"],
            "state": t["state"],
            "commodity": t["commodity"],
            "mandi_hi": t.get("market_hi") or t["market"],
            "state_hi": t.get("state_hi") or t["state"],
        }

    @property
    def target_slug(self) -> str:
        """A filesystem-safe `<market>_<commodity>` label for output file names.

        So the Time-LLM export is named after whichever market is configured, instead of a
        hardcoded "indore_soyabean".

        Example:  s.target_slug  ->  'indore_soyabean'   (for market=Indore, commodity=Soyabean)
        """
        t = self.raw["target"]
        raw = f"{t['market']}_{t['commodity']}".lower()
        return "".join(c if c.isalnum() else "_" for c in raw).strip("_")

    @property
    def work_dir(self) -> Path:
        """Per-target working folder: `data/<target_slug>/` (e.g. data/indore_soyabean).

        Everything the pipeline generates — the raw cache, staged tables, features and the
        agent progress file — lives under here, keyed by the configured market. So switching
        `target.market` in settings.yaml puts the new market's data in its OWN folder and can
        never overwrite another market's staged tables or corrupt its features. The manual
        Agmarknet input folder (`mandi_sources.csv.import_dir`) is deliberately NOT under here:
        it is user-supplied input, read directly, and the mandi cleaner filters it by market.

        Example:  s.work_dir  ->  WindowsPath('.../data/indore_soyabean')
        """
        return self.data_dir / self.target_slug

    def path(self, *parts: str) -> Path:
        """Build a path inside the per-target working folder and create its parent folders.

        Example: s.path("staged", "mandi_daily.parquet")
                 -> <root>/data/indore_soyabean/staged/mandi_daily.parquet  (parents created if missing)
        """
        p = self.work_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_settings(path: str | Path | None = None) -> Settings:
    """Load .env and settings.yaml and return a Settings object.

    path : optional alternative YAML file (CLI flag --config). Defaults to config/settings.yaml.

    Example:
        load_settings()                  -> Settings(raw={'project': {...}, 'target': {...}, ...}, root=.../soyfc)
        load_settings("my_config.yaml")  -> Settings built from my_config.yaml instead
    """
    root = ROOT
    # 1) Put the keys from <root>/.env into os.environ (does nothing if .env is absent).
    if load_dotenv:
        load_dotenv(root / ".env")
    # 2) Parse the YAML settings file.
    cfg_path = Path(path) if path else root / "config" / "settings.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)  # safe_load: plain data only, never executes YAML tags
    return Settings(raw=raw, root=root)


def env(name: str, required: bool = False) -> str | None:
    """Read an API key / secret from the environment.

    name     : variable name, e.g. "TAVILY_API_KEY"
    required : if True, stop with a clear error when the key is missing or empty,
               instead of failing later with a confusing HTTP 401.
    Returns the value, or None when it's missing/empty and not required.

    Example:
        env("TAVILY_API_KEY")                 -> 'tvly-xxxxxxxx'   (or None if unset)
        env("MISSING_KEY", required=True)     -> raises RuntimeError("Missing environment variable ...")
    """
    val = os.getenv(name) or None  # treat "" (KEY= with no value) the same as missing
    if required and not val:
        raise RuntimeError(f"Missing environment variable {name}. Add it to .env (see .env.example).")
    return val