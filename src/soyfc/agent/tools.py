"""Window-bound research tools for the LangChain agent.

The agent decides WHAT to search for and save; these tools decide what is ALLOWED. Every rule
that protects the training data from look-ahead and hallucination lives here, in code, not in
the prompt:

  search_news(query)   Tavily search pinned to the window's dates. Articles published after the
                       window are hidden. Each run has a hard search budget. Responses are cached.
  read_doc(doc_id)     full text of a document a search returned.
  save(records)        validates and keeps records. A record is rejected unless:
                         - its doc_id is a document the agent actually retrieved,
                         - its quote appears verbatim in that document's text,
                         - confidence >= settings.agent.min_confidence,
                         - the document was published within the window (+3 days grace),
                         - any obs_date it carries is not after the window.
                       The tool sets `available_at` itself; the agent cannot.

State for one research run lives in a single `Window` (one area, one time window). The tools are
plain functions closed over that Window; nothing here is a class with behaviour of its own.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from .. import store
from ..config import Settings
from .tasks import TASKS

log = logging.getLogger(__name__)
GRACE_DAYS = 3   # articles published up to 3 days after the window still describe it


# =========================================================================== helpers
def _to_date(x: Any) -> pd.Timestamp | None:
    """Parse any date-like value to a timezone-naive day; None if missing/unparseable.
    Tavily's published_date may be ISO or RFC-822 ("Mon, 04 Nov 2024 10:00:00 GMT").

    Example:
        _to_date("Mon, 04 Nov 2024 10:00:00 GMT")  -> Timestamp('2024-11-04')
        _to_date("2024-11-06")                     -> Timestamp('2024-11-06')
        _to_date(None)                             -> None
    """
    if x in (None, "", "null"):
        return None
    try:
        t = pd.to_datetime(x, errors="coerce", utc=True)
        return None if pd.isna(t) else t.tz_convert(None).normalize()
    except Exception:  # noqa: BLE001
        return None


def normalize_text(t: str) -> str:
    """Normalise text for the verbatim-quote check.

    NFKC unifies Unicode variants (e.g. Devanagari forms), lower-casing ignores case, and every
    non-word character becomes a space, so "4,950" vs "4950"-style punctuation, curly quotes and
    line breaks don't cause false rejections. The same function is applied to quote and document,
    so the check stays exact on the words themselves (Hindi included).

    Example:
        normalize_text("Rs 4,950/qtl (modal)")  -> 'rs 4 950 qtl modal'
        # so a quote "Rs 4950 qtl" would still match the document text "Rs 4,950/qtl"
    """
    t = unicodedata.normalize("NFKC", t or "").lower()
    t = re.sub(r"[^\w]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class SearchInput(BaseModel):
    query: str = Field(description="search query in English or Hindi")


class ReadInput(BaseModel):
    doc_id: str = Field(description="doc_id from a search_news result, e.g. 'd3'")


# =========================================================================== window state
@dataclass
class Window:
    """Everything one research run (one area, one time window) accumulates.

    Created fresh per run and discarded afterwards apart from `saved`, which the runner stages.

    Example:
        win = Window(task="news_sentiment", start=pd.Timestamp("2024-11-01"),
                     end=pd.Timestamp("2024-11-30"), budget=6, min_confidence=0.5)
        # after the agent runs: win.searches==3, win.saved==[{...}, {...}], win.rejected==Counter({...})
    """

    task: str                                       # research area name, e.g. "news_sentiment"
    start: pd.Timestamp
    end: pd.Timestamp
    budget: int                                     # max search_news calls allowed this run
    min_confidence: float
    docs: dict[str, dict] = field(default_factory=dict)      # doc_id -> {url, title, text, ...}
    url_to_id: dict[str, str] = field(default_factory=dict)  # same URL twice -> same doc_id
    saved: list[dict] = field(default_factory=list)          # validated records to stage
    rejected: Counter = field(default_factory=Counter)       # reason -> count
    searches: int = 0                               # search_news calls made
    search_errors: int = 0                          # failed Tavily calls


def window_frame(win: Window) -> pd.DataFrame:
    """Saved records as a DataFrame (empty if none were kept).

    Example:  window_frame(win)  ->  DataFrame[doc_id, quote, sentiment, available_at, url, task, ...], 2 rows
    """
    return pd.DataFrame(win.saved)


def window_stats(win: Window, agent_error: str | None = None) -> dict:
    """One-line summary of a run, for the runner's progress table.

    Example:
        window_stats(win)  -> {"task": "news_sentiment", "docs": 5, "searches": 3, "search_errors": 0,
                               "kept": 2, "rejected": 1, "agent_error": None}
    """
    return {"task": win.task, "docs": len(win.docs), "searches": win.searches,
            "search_errors": win.search_errors, "kept": len(win.saved),
            "rejected": int(sum(win.rejected.values())), "agent_error": agent_error}


# =========================================================================== tool bodies
def _search(s: Settings, tavily, win: Window, query: str) -> str:
    """Tavily search for this window. Returns a JSON listing (doc_ids, titles, snippets).

    Example:
        _search(s, tavily, win, "सोयाबीन भाव इंदौर")
          -> '{"results": [{"doc_id": "d1", "title": "...", "url": "...", "published": "2024-11-06",
                            "snippet": "..."}], "searches_left": 5}'
          -> '{"error": "search budget exhausted ..."}'   once win.budget searches are used
    """
    if win.searches >= win.budget:
        return json.dumps({"error": "search budget exhausted for this window; save what you have and finish"})
    win.searches += 1
    cfg, t = s["agent"], TASKS[win.task]
    n = cfg["max_results_per_search"]
    key = f"{t.topic}|{query}|{win.start.date()}|{win.end.date()}|{n}"
    cached = store.cache_get_json(s, "tavily", key)
    if cached:
        res = cached["data"]
    else:
        try:
            res = tavily.search(
                query=query, topic=t.topic, search_depth=cfg["tavily_search_depth"], max_results=n,
                include_raw_content=True,                                   # full text for read_doc / quote check
                days=None,                                                  # the news topic defaults days=7, which
                                                                            # Tavily rejects alongside a date window
                start_date=str(win.start.date()), end_date=str(win.end.date()),   # window pin (code, not prompt)
                **({"include_domains": t.include_domains} if t.include_domains else {}))
        except Exception as e:  # noqa: BLE001 - credits, rate limit, network
            log.warning("tavily failed for %r: %s", query, e)
            win.search_errors += 1
            return json.dumps({"error": f"search failed: {e}"})
        store.cache_put_json(s, "tavily", key, res)
    listing = []
    for r in res.get("results", []):
        pub = _to_date(r.get("published_date"))
        if pub is not None and pub > win.end + pd.Timedelta(days=GRACE_DAYS):
            continue                     # hide later articles entirely: they could leak the future
        url = r["url"]
        if url not in win.url_to_id:
            did = f"d{len(win.docs) + 1}"
            win.url_to_id[url] = did
            win.docs[did] = {"url": url, "title": r.get("title", ""), "published_date": r.get("published_date"),
                             "text": r.get("raw_content") or r.get("content") or "", "query": query}
        d = win.docs[win.url_to_id[url]]
        listing.append({"doc_id": win.url_to_id[url], "title": d["title"], "url": url,
                        "published": d["published_date"], "snippet": (r.get("content") or d["text"])[: cfg["snippet_chars"]]})
    return json.dumps({"results": listing, "searches_left": win.budget - win.searches}, ensure_ascii=False)


def _read(s: Settings, win: Window, doc_id: str) -> str:
    """Full (truncated) text of a retrieved document.

    Example:
        _read(s, win, "d1")  -> '{"doc_id": "d1", "title": "...", "url": "...", "published": "2024-11-06",
                                  "text": "<full article text, truncated to raw_content_chars>"}'
        _read(s, win, "d9")  -> '{"error": "unknown doc_id d9; use ids from search_news results"}'
    """
    d = win.docs.get(doc_id)
    if not d:
        return json.dumps({"error": f"unknown doc_id {doc_id}; use ids from search_news results"})
    return json.dumps({"doc_id": doc_id, "title": d["title"], "url": d["url"], "published": d["published_date"],
                       "text": d["text"][: s["agent"]["raw_content_chars"]]}, ensure_ascii=False)


def _check(win: Window, rec: dict) -> tuple[dict | None, str | None]:
    """Validate one record. Returns (row to store, None) or (None, rejection reason).

    Example:
        _check(win, {"doc_id": "d1", "quote": "Soybean prices rose Rs 100 in Indore mandi",
                     "confidence": 0.9, "headline": "h", "relevance": 0.9, "sentiment": 0.6})
          -> ({..., "url": "https://...", "available_at": Timestamp('2024-11-06'), "task": "news_sentiment"}, None)
        _check(win, {... "quote": "text not in the article"})
          -> (None, "quote not found verbatim in the document (...)")
    """
    try:
        rec = TASKS[win.task].record_model.model_validate(rec).model_dump()
    except ValidationError as e:
        return None, f"invalid fields: {e.errors()[0]['msg']}"
    d = win.docs.get(str(rec.get("doc_id", "")).strip())
    if d is None:
        return None, "doc_id not retrieved in this window"
    q = normalize_text(rec.get("quote", ""))
    if len(q) < 8:
        return None, "quote too short (copy a phrase of several words)"
    if q not in normalize_text(d["text"]):
        return None, "quote not found verbatim in the document (copy it exactly, original language)"
    if rec.get("confidence", 0) < win.min_confidence:
        return None, "confidence below threshold"
    # Publish date: search engine's first, else the date the agent read in the text.
    pub = _to_date(d.get("published_date")) or _to_date(rec.get("published_at"))
    if pub is not None and pub > win.end + pd.Timedelta(days=GRACE_DAYS):
        return None, "document published after the window"
    obs = _to_date(rec.get("obs_date"))
    if obs is not None and obs > win.end + pd.Timedelta(days=GRACE_DAYS):
        return None, "obs_date is after the window"
    # available_at: publish date, or window end if unknown; never before the fact itself.
    avail = pub if pub is not None else win.end
    facts = [x for x in (_to_date(rec.get("announced_at")), obs) if x is not None]
    if facts and avail < max(facts):
        avail = max(facts)
    row = {**rec, "url": d["url"], "title": d["title"], "query": d["query"],
           "published_at": pub, "available_at": avail,
           "retrieved_at": pd.Timestamp(datetime.now(timezone.utc).date()),
           "window_start": str(win.start.date()), "task": win.task}
    # features/build.py reads list-valued fields (e.g. news topics) as pipe-joined strings.
    return {k: "|".join(map(str, v)) if isinstance(v, list) else v for k, v in row.items()}, None


def _save(win: Window, records: list) -> str:
    """Validate and keep records; tell the agent what was rejected and why, so it can fix them.

    Example:
        _save(win, [{good record}, {bad quote record}])
          -> '{"saved": 1, "rejected": {"quote not found verbatim in the document (...)": 1},
               "total_saved_this_window": 1}'
    """
    kept, why = 0, Counter()
    for r in records:
        row, reason = _check(win, r if isinstance(r, dict) else r.model_dump())
        if row is None:
            why[reason] += 1
            win.rejected[reason] += 1
        else:
            win.saved.append(row)
            kept += 1
    return json.dumps({"saved": kept, "rejected": dict(why), "total_saved_this_window": len(win.saved)})


# =========================================================================== tool objects
def make_tools(s: Settings, tavily, win: Window) -> list[StructuredTool]:
    """Build search_news / read_doc / save_<area> bound to one research run.

    Example:
        search, read, save = make_tools(s, tavily, win)   # win.task == "news_sentiment"
        search.name  -> 'search_news';  save.name  -> 'save_news_sentiment'
        # each is a langchain StructuredTool the agent calls, e.g. search.invoke({"query": "..."})
    """
    t = TASKS[win.task]
    return [
        StructuredTool.from_function(
            func=lambda query: _search(s, tavily, win, query), name="search_news", args_schema=SearchInput,
            description=f"Search news/web for this window only ({win.start.date()} to {win.end.date()}). "
                        f"Budget: {win.budget} searches. Returns doc_ids, titles, snippets."),
        StructuredTool.from_function(
            func=lambda doc_id: _read(s, win, doc_id), name="read_doc", args_schema=ReadInput,
            description="Read the full text of a document returned by search_news."),
        StructuredTool.from_function(
            func=lambda records: _save(win, records), name=f"save_{win.task}", args_schema=t.schema,
            description=f"Save {win.task} records. Each needs a doc_id and a quote copied verbatim from that "
                        "document. Returns how many were saved and why any were rejected.",
            handle_validation_error=True),
    ]
