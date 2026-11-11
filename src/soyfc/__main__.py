"""Command-line entry point. Run from the project folder with src on PYTHONPATH:

    python -m soyfc <command>

  collect  [mandi|weather|markets|psd|all]   structured sources -> data/staged
  agent    --tasks ... [--start --end] [--dry-run] [--update] [--force]
                                            Tavily + LLM research agent -> data/staged
  build                                     weekly features + training data -> data/features
  status                                    staged tables: row counts and date ranges
  run-all                                   weekly job: collect all + agent --update + build
  probe-datagov                             show data.gov.in resource fields (verify ids)
  probe-mandi-api                           show mandi-api coverage + sample rows for the target mandi

`python -m soyfc` works because Python runs this file (__main__.py) when a package is
executed with -m.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta

import pandas as pd

from .config import load_settings

# Collector names accepted by `collect`, in the order `collect all` runs them.
COLLECTORS = ["mandi", "weather", "markets", "psd"]
# CLI name -> module in soyfc/collectors/ when they differ.
MODULE_FOR = {"psd": "usda_psd"}
# Research tasks run by `agent` (when --tasks is not given) and by `run-all`.
ALL_TASKS = ["policy_events", "industry_stats", "product_prices", "news_sentiment", "price_forecasts"]


def main(argv=None):
    """Parse the command line and dispatch to collect / agent / build / status / probes.

    argv : argument list (defaults to sys.argv[1:]); passing a list is handy for tests.

    Example:
        main(["collect", "weather"])                 # stage weather_daily, prints row count
        main(["agent", "--update"])                  # research the last 7 days, one window
        main(["build"])                              # write data/features/*, prints the result JSON
        main(["status"])                             # list staged tables with row counts + date ranges
    """
    # ------------------------------------------------------------------ arguments
    ap = argparse.ArgumentParser(prog="soyfc")
    ap.add_argument("--config", default=None, help="alternative settings YAML (default config/settings.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="run structured collectors")
    c.add_argument("what", nargs="*", default=["all"], help="any of: mandi weather markets psd all")

    a = sub.add_parser("agent", help="run the Tavily + LLM research agent")
    a.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    a.add_argument("--start", help="backfill start date (default: project start_date)")
    a.add_argument("--end", help="backfill end date (default: today)")
    a.add_argument("--update", action="store_true", help="only the last update_window_days, one window")
    a.add_argument("--dry-run", action="store_true", help="show planned windows and Tavily searches, spend nothing")
    a.add_argument("--force", action="store_true", help="re-run windows already marked done")

    sub.add_parser("build", help="build weekly features + training files")
    sub.add_parser("status", help="list staged tables")
    sub.add_parser("probe-datagov", help="inspect data.gov.in resources")
    sub.add_parser("probe-mandi-api", help="inspect mandi-api coverage for the target mandi")
    sub.add_parser("run-all", help="collect all + agent --update + build (weekly job)")
    args = ap.parse_args(argv)

    # Log to stdout with timestamps, so a Task Scheduler run can be redirected to a log file.
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    s = load_settings(args.config)

    # ------------------------------------------------------------------ collect
    if args.cmd in ("collect", "run-all"):
        what = COLLECTORS if args.cmd == "run-all" or "all" in getattr(args, "what", ["all"]) else args.what
        for w in what:
            # Import soyfc.collectors.<module> by name; every collector exposes collect(s).
            mod = __import__(f"soyfc.collectors.{MODULE_FOR.get(w, w)}", fromlist=["collect"])
            try:
                df = mod.collect(s)
                logging.info("collect %s -> %d rows staged", w, len(df))
            except Exception as e:  # noqa: BLE001 - one failing source shouldn't stop the others
                logging.exception("collect %s failed: %s", w, e)

    # ------------------------------------------------------------------ agent
    if args.cmd in ("agent", "run-all"):
        from .agent import runner   # imported lazily: collect/build don't need LangChain installed

        upd = args.cmd == "run-all" or args.update
        if upd:
            # Weekly update: a single window covering the last N days (default 7).
            end = date.today()
            start = end - timedelta(days=s["agent"]["update_window_days"] - 1)
            freq = f"{s['agent']['update_window_days']}D"
        else:
            # Backfill: monthly windows (settings.agent.backfill_window) from start to end.
            start, end, freq = (args.start or s.start_date), (args.end or s.end_date), None
        tasks = args.tasks if args.cmd == "agent" else ALL_TASKS
        # Updates always re-run (force) because the latest week's window is never "complete".
        res = runner.run(s, tasks, str(start), str(end), freq=freq,
                         force=getattr(args, "force", False) or upd, dry_run=getattr(args, "dry_run", False))
        if getattr(args, "dry_run", False):
            todo = res[~res["done"]]
            print(todo.groupby("task").agg(windows=("window_start", "size"), max_searches=("max_searches", "sum")).to_string())
            # Upper bound: the search tool refuses once an area's per-window budget is used.
            print(f"\nAt most {int(todo['max_searches'].sum())} Tavily searches")
        else:
            print(res.to_string() if not res.empty else "nothing to do")

    # ------------------------------------------------------------------ build
    if args.cmd in ("build", "run-all"):
        from .features.build import build_all
        print(json.dumps(build_all(s), indent=2))

    # ------------------------------------------------------------------ status
    if args.cmd == "status":
        # s.path(...) creates data/staged if needed; .parent is the folder itself.
        for p in sorted(s.path("staged", "x").parent.glob("*.parquet")):
            df = pd.read_parquet(p)
            rng = (f"{pd.to_datetime(df['obs_date']).min().date()} → {pd.to_datetime(df['obs_date']).max().date()}"
                   if "obs_date" in df and len(df) else "")
            print(f"{p.stem:28s} {len(df):>8d} rows  {rng}")

    # ------------------------------------------------------------------ probes
    if args.cmd == "probe-datagov":
        from .collectors.mandi import probe_datagov
        print(json.dumps(probe_datagov(s), indent=2, default=str))
    if args.cmd == "probe-mandi-api":
        from .collectors.mandi import probe_mandi_api
        # ensure_ascii=False so Hindi market names print readably
        print(json.dumps(probe_mandi_api(s), indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()