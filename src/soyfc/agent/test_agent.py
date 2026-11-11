"""Research agent tests — no network, no API keys.

1. The code-enforced tool rules (tools.py): window pinning, hiding later articles, the search
   budget, the verbatim-quote check and available_at.
2. The runner loop (runner.py) with a fake researcher: windows, staging, progress tracking.
3. A failed search leaves the window not-done, so it is retried next run.
4. The real LangChain agent (agent.py) end-to-end with a scripted fake chat model (no network).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from soyfc import store  # noqa: E402
from soyfc.agent import agent as agent_mod  # noqa: E402
from soyfc.agent import runner  # noqa: E402
from soyfc.agent.tools import Window, make_tools  # noqa: E402
from soyfc.config import load_settings  # noqa: E402

ARTICLE = ("इंदौर मंडी में सोयाबीन के भाव 100 रुपये बढ़े। Soybean prices rose Rs 100 in Indore mandi on lower arrivals, "
           "traders said on Wednesday.")


class FakeTavily:
    """Returns one in-window article and one later (future) article that must be hidden."""

    calls = []

    def search(self, query, **kw):
        FakeTavily.calls.append(kw)
        return {"results": [
            {"url": "https://example.com/a", "title": "Soybean prices rise in Indore", "content": ARTICLE[:80],
             "raw_content": ARTICLE, "published_date": "2024-11-06"},
            {"url": "https://example.com/late", "title": "Future article", "content": "x", "raw_content": "x",
             "published_date": "2025-03-01"},
        ]}


class FailingTavily:
    def search(self, query, **kw):
        raise RuntimeError("credits exhausted")


def settings(tmp_path):
    s = load_settings()
    s.raw["project"]["data_dir"] = str(tmp_path)
    s.raw["agent"]["max_searches_per_task"] = 2
    return s


def _window(s, start="2024-11-01", end="2024-11-30"):
    return Window(task="news_sentiment", start=pd.Timestamp(start), end=pd.Timestamp(end),
                  budget=s["agent"]["max_searches_per_task"], min_confidence=s["agent"]["min_confidence"])


# ---------------------------------------------------------------------------------- tools
def test_tool_rules(tmp_path):
    s = settings(tmp_path)
    win = _window(s)
    search, read, save = make_tools(s, FakeTavily(), win)

    out = json.loads(search.invoke({"query": "soybean Indore"}))
    assert [r["doc_id"] for r in out["results"]] == ["d1"]                 # later article hidden
    assert FakeTavily.calls[-1]["start_date"] == "2024-11-01" and FakeTavily.calls[-1]["end_date"] == "2024-11-30"
    assert "Soybean prices rose" in json.loads(read.invoke({"doc_id": "d1"}))["text"]

    base = {"doc_id": "d1", "published_at": None, "confidence": 0.9, "headline": "h", "relevance": 0.9, "sentiment": 0.5}
    res = json.loads(save.invoke({"records": [
        {**base, "quote": "Soybean prices rose Rs 100 in Indore mandi"},          # ok (English)
        {**base, "quote": "इंदौर मंडी में सोयाबीन के भाव 100 रुपये बढ़े"},           # ok (Hindi, verbatim)
        {**base, "quote": "prices crashed across Madhya Pradesh"},              # not in document
        {**base, "doc_id": "d9", "quote": "Soybean prices rose Rs 100"},        # never retrieved
        {**base, "quote": "Soybean prices rose Rs 100", "confidence": 0.1},     # low confidence
    ]}))
    assert res["saved"] == 2 and sum(res["rejected"].values()) == 3
    row = win.saved[0]
    assert row["available_at"] == pd.Timestamp("2024-11-06") and row["url"] == "https://example.com/a"

    search.invoke({"query": "second query"})
    assert "budget exhausted" in search.invoke({"query": "third query"})      # budget = 2


# ---------------------------------------------------------------------------------- runner loop
def test_runner_stages_and_tracks(tmp_path):
    s = settings(tmp_path)

    def researcher(task, ws, we):
        """Stand-in for the LLM agent: save one record for the window."""
        df = pd.DataFrame([{"task": task, "available_at": pd.Timestamp("2024-11-06"), "topics": "supply",
                            "url": "https://example.com/a", "headline": "h", "relevance": 0.9, "sentiment": 0.6}])
        return df, {"task": task, "kept": 1, "rejected": 1, "searches": 1, "search_errors": 0, "agent_error": None}

    out = runner.run(s, ["news_sentiment"], "2024-11-01", "2024-11-30", researcher=researcher)
    assert len(out) == 1 and out.iloc[0]["complete"] and out.iloc[0]["kept"] == 1 and out.iloc[0]["rejected"] == 1
    assert len(store.read(s, "agent_news_sentiment")) == 1
    assert runner.plan(s, ["news_sentiment"], "2024-11-01", "2024-11-30", "MS")["done"].all()


def test_failed_window_not_marked_done(tmp_path):
    s = settings(tmp_path)

    def failing_researcher(task, ws, we):
        """A search that errors: nothing saved, search_errors recorded -> window not complete."""
        win = _window(s)
        make_tools(s, FailingTavily(), win)[0].invoke({"query": "anything"})
        return pd.DataFrame(), {"task": task, "kept": 0, "rejected": 0, "searches": 1,
                                "search_errors": win.search_errors, "agent_error": None}

    out = runner.run(s, ["news_sentiment"], "2024-11-01", "2024-11-30", researcher=failing_researcher)
    assert not out.iloc[0]["complete"] and out.iloc[0]["search_errors"] == 1
    assert not runner.plan(s, ["news_sentiment"], "2024-11-01", "2024-11-30", "MS")["done"].any()


def test_dry_run_plan(tmp_path):
    s = settings(tmp_path)
    p = runner.run(s, list(runner.TASKS), "2015-01-01", "2026-10-31", dry_run=True)
    assert p["window_start"].nunique() == 142 and len(p) == 142 * 5


# ---------------------------------------------------------------------------------- real agent, fake model
def test_research_window_with_scripted_model(tmp_path):
    """research_window drives the real LangChain agent; only the chat model is faked."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class ScriptedModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    s = settings(tmp_path)
    model = ScriptedModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "search_news", "id": "s1", "args": {"query": "सोयाबीन भाव इंदौर"}}]),
        AIMessage(content="", tool_calls=[{"name": "read_doc", "id": "r1", "args": {"doc_id": "d1"}}]),
        AIMessage(content="", tool_calls=[{"name": "save_news_sentiment", "id": "v1", "args": {"records": [
            {"doc_id": "d1", "quote": "Soybean prices rose Rs 100 in Indore mandi", "published_at": "2024-11-06",
             "confidence": 0.9, "headline": "Soybean prices rise in Indore", "relevance": 0.9, "sentiment": 0.6,
             "topics": ["supply"]},
            {"doc_id": "d1", "quote": "a sentence that is not in the article", "confidence": 0.9,
             "headline": "x", "relevance": 0.5, "sentiment": -0.2}]}}]),
        AIMessage(content="Saved 1 record; nothing else relevant."),
    ]))
    df, stats = agent_mod.research_window(s, FakeTavily(), model, "news_sentiment", "2024-11-01", "2024-11-30")
    assert stats["kept"] == 1 and stats["rejected"] == 1 and stats["agent_error"] is None
    assert len(df) == 1 and df.iloc[0]["topics"] == "supply"   # list serialized to pipe-joined string
    assert pd.Timestamp(df.iloc[0]["available_at"]) == pd.Timestamp("2024-11-06")
