"""The research agent: one standard LangChain tool-calling agent per area, per window.

We build a single agent (`langchain.agents.create_agent`) with three tools — search_news,
read_doc and save_<area> — and let it loop (search -> read -> save) until it is done or runs
out of its search budget. One area per agent keeps each prompt focused; there is no orchestrator
or subagent machinery — the runner (runner.py) just calls `research_window` for each area and
window in a plain Python loop.

What the prompt can NOT override (enforced in tools.py): the window dates, the search budget,
the verbatim-quote check and `available_at`.
"""
from __future__ import annotations

import logging

import pandas as pd
from langchain.agents import create_agent

from ..config import Settings
from .tasks import COMMON, TASKS, render_hints
from .tools import Window, make_tools, window_frame, window_stats

log = logging.getLogger(__name__)

RESEARCHER_PROMPT = """You are the `{name}` researcher for a {mandi} ({state}) soybean \
price-forecasting dataset. Research window: {start} to {end}. Your tools only return documents \
published in this window.

Goal: {instructions}

{common}

Example searches (adapt and vary them; mix Hindi and English):
{hints}

How to work:
1. search_news with a focused query; look at titles, dates and snippets.
2. read_doc on promising documents before saving anything from them.
3. save_{name}(records=[...]) — each record with the doc_id and an exact quote from that document. \
If the save result lists rejections, fix those records (usually the quote) and save again.
4. Search again with different wording or language while useful results keep coming. You have \
{budget} searches; stop when they run out or results stop being relevant.
5. Reply with a short summary: how many records you saved and what you could not find."""


def build_llm(s: Settings):
    """The chat model for the agent, from settings.agent.model (any init_chat_model id).
    The Gemini client reads GOOGLE_API_KEY from the environment.

    Example:
        build_llm(s)  ->  ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(s["agent"]["model"], temperature=0)


def research_window(s: Settings, tavily, llm, task: str, start, end) -> tuple[pd.DataFrame, dict]:
    """Research one area for one window. Returns (saved records, stats).

    Records saved before any mid-run failure are still returned, and `agent_error` is set in the
    stats so the runner retries the window next time.

    Example:
        df, stats = research_window(s, tavily, llm, "news_sentiment", "2024-11-01", "2024-11-30")
        df     -> DataFrame of saved news records (0+ rows)
        stats  -> {"task": "news_sentiment", "docs": 5, "searches": 3, "search_errors": 0,
                   "kept": 2, "rejected": 1, "agent_error": None}
    """
    win = Window(task=task, start=pd.Timestamp(start), end=pd.Timestamp(end),
                 budget=s["agent"]["max_searches_per_task"], min_confidence=s["agent"]["min_confidence"])
    t = TASKS[task]
    # market names from settings, so prompt/hints/instructions name the configured mandi, not Indore.
    terms = s.market_terms()
    hints = "\n".join(f"- {h}" for h in render_hints(t, win.start, terms))
    prompt = RESEARCHER_PROMPT.format(name=task, start=win.start.date(), end=win.end.date(),
                                      instructions=t.instructions.format(**terms), common=COMMON,
                                      hints=hints, budget=win.budget, mandi=terms["mandi"], state=terms["state"])
    agent = create_agent(model=llm, tools=make_tools(s, tavily, win), system_prompt=prompt)
    error = None
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": f"Research {task} for the window {win.start.date()} to {win.end.date()}."}]},
            config={"recursion_limit": s["agent"]["recursion_limit"]})
    except Exception as e:  # noqa: BLE001 - model error / step limit: keep what was saved
        error = f"{type(e).__name__}: {e}"
        log.warning("agent failed for %s %s..%s: %s", task, win.start.date(), win.end.date(), error)
    return window_frame(win), window_stats(win, error)
