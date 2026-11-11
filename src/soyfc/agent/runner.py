"""Run the research agent across time windows, with progress tracking and retries.

Deterministic Python owns the loop here; the LLM only researches inside a single window
(agent.py). This module:

  windows(...)  split a date range into windows (monthly for backfill, one window for updates).
  plan(...)     the full (area x window) grid, with a `done` flag from the progress file.
  run(...)      research every (area, window) that isn't done yet, stage the saved records, and
                record progress so re-runs skip finished work and retry failed windows.

Progress is a small table at data/agent_progress.parquet, keyed by (task, window_start). A window
counts as done only when the agent finished without error and no Tavily search failed — so a run
cut short by exhausted credits is retried next time rather than silently left half-filled.
"""
from __future__ import annotations

import logging

import pandas as pd

from .. import store
from ..config import Settings, env
from .tasks import TASKS   # re-exported: callers use runner.TASKS

log = logging.getLogger(__name__)

__all__ = ["TASKS", "windows", "plan", "run"]


# =========================================================================== windows
def windows(start, end, freq: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end] into (window_start, window_end) pairs at the given pandas frequency.

    freq "MS" gives one window per calendar month (backfill); a span shorter than one period
    gives a single window covering the whole range (the weekly update case).

    Example:
        windows("2024-09-01", "2024-11-30", "MS")
          -> [(2024-09-01, 2024-09-30), (2024-10-01, 2024-10-31), (2024-11-01, 2024-11-30)]
        windows("2024-11-24", "2024-11-30", "7D")   -> [(2024-11-24, 2024-11-30)]   # one update window
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    starts = pd.date_range(start, end, freq=freq)
    if len(starts) == 0:
        return [(start, end)]
    out = []
    for i, ws in enumerate(starts):
        we = starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(starts) else end
        out.append((ws, min(we, end)))
    return out


# =========================================================================== progress
def _progress_path(s: Settings):
    return s.path("agent_progress.parquet")


def _load_progress(s: Settings) -> pd.DataFrame:
    """The progress table (empty with the right columns if no run has happened yet).

    Example:  _load_progress(s)  ->  DataFrame[task, window_start, window_end, complete, kept, ...]
    """
    p = _progress_path(s)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["task", "window_start", "complete"])


def _record(s: Settings, row: dict) -> None:
    """Upsert one window's result into the progress file (keep the latest per task+window).

    Example:
        _record(s, {"task": "news_sentiment", "window_start": "2024-11-01", "complete": True, "kept": 2})
          -> rewrites data/agent_progress.parquet with this row added/replaced
    """
    prog = pd.concat([_load_progress(s), pd.DataFrame([row])], ignore_index=True)
    prog = prog.drop_duplicates(["task", "window_start"], keep="last")
    prog.to_parquet(_progress_path(s), index=False)


def _done_set(s: Settings) -> set[tuple[str, str]]:
    """(task, window_start) pairs already completed successfully.

    Example:  _done_set(s)  ->  {("news_sentiment", "2024-11-01"), ("policy_events", "2024-10-01")}
    """
    prog = _load_progress(s)
    done = prog[prog["complete"] == True] if not prog.empty else prog  # noqa: E712
    return set(zip(done["task"], done["window_start"]))


# =========================================================================== plan / run
def plan(s: Settings, tasks: list[str], start, end, freq: str | None = None) -> pd.DataFrame:
    """The whole (task x window) grid with a `done` flag. freq defaults to the backfill window.

    Example:
        plan(s, ["news_sentiment"], "2024-09-01", "2024-11-30", "MS")
          ->          task        window_start window_end   done
             news_sentiment        2024-09-01  2024-09-30   True
             news_sentiment        2024-10-01  2024-10-31  False
             news_sentiment        2024-11-01  2024-11-30  False
    """
    freq = freq or s["agent"]["backfill_window"]
    done = _done_set(s)
    rows = [{"task": t, "window_start": str(ws.date()), "window_end": str(we.date()),
             "done": (t, str(ws.date())) in done}
            for ws, we in windows(start, end, freq) for t in tasks]
    return pd.DataFrame(rows)


def _default_researcher(s: Settings):
    """A `research(task, start, end) -> (DataFrame, stats)` callable backed by Tavily + the LLM.
    Built lazily so `plan`/`dry_run` never need API keys.

    Example:
        research = _default_researcher(s)
        df, stats = research("news_sentiment", pd.Timestamp("2024-11-01"), pd.Timestamp("2024-11-30"))
    """
    from tavily import TavilyClient

    from . import agent as agent_mod
    tavily = TavilyClient(api_key=env("TAVILY_API_KEY", required=True))
    llm = agent_mod.build_llm(s)
    return lambda task, ws, we: agent_mod.research_window(s, tavily, llm, task, ws, we)


def run(s: Settings, tasks: list[str], start, end, freq: str | None = None,
        force: bool = False, dry_run: bool = False, researcher=None) -> pd.DataFrame:
    """Research `tasks` across the windows between `start` and `end`.

    freq       : window frequency (default settings.agent.backfill_window, "MS"); "7D" etc. for updates.
    force      : re-run windows already marked done.
    dry_run    : return the plan (with an upper-bound search count) without spending anything.
    researcher : a `(task, start, end) -> (DataFrame, stats)` callable; tests inject a fake. When
                 None, a Tavily + LLM researcher is built from settings.
    Returns one row per window actually run (or the whole plan for a dry run).

    Example:
        # dry run (no API calls): the plan with an upper-bound search count
        run(s, ["news_sentiment"], "2024-09-01", "2024-11-30", dry_run=True)
          -> DataFrame[task, window_start, window_end, done, max_searches]
        # real run: researches each not-yet-done window, stages agent_<task>, records progress
        run(s, ["news_sentiment"], "2024-11-01", "2024-11-30")
          -> DataFrame[task, window_start, window_end, complete, kept, rejected, searches, search_errors]
    """
    freq = freq or s["agent"]["backfill_window"]
    budget = s["agent"]["max_searches_per_task"]

    if dry_run:
        p = plan(s, tasks, start, end, freq)
        p["max_searches"] = (~p["done"]).astype(int) * budget
        return p

    if researcher is None:
        researcher = _default_researcher(s)
    done = set() if force else _done_set(s)

    rows = []
    for ws, we in windows(start, end, freq):
        for task in tasks:
            if (task, str(ws.date())) in done:
                continue
            df, stats = researcher(task, ws, we)
            if df is not None and not df.empty:
                store.upsert(s, f"agent_{task}", df, TASKS[task].keys)
            complete = stats.get("agent_error") is None and stats.get("search_errors", 0) == 0
            row = {"task": task, "window_start": str(ws.date()), "window_end": str(we.date()),
                   "complete": complete, "kept": stats.get("kept", 0), "rejected": stats.get("rejected", 0),
                   "searches": stats.get("searches", 0), "search_errors": stats.get("search_errors", 0)}
            _record(s, row)
            rows.append(row)
            log.info("agent %s %s..%s: kept=%s rejected=%s complete=%s",
                     task, ws.date(), we.date(), row["kept"], row["rejected"], complete)
    return pd.DataFrame(rows)
