"""Research areas: one specialist subagent per area.

A ResearchTask bundles everything that differs between areas:
  schema        wrapper model `{records: [...]}` — becomes the argument schema of the area's
                save tool (save_<name>), so the agent can only save records of this shape
  record_model  the single-record model, used to re-validate each record individually
  description   one line shown to the orchestrator when it picks a subagent
  instructions  what the subagent must find (goes into its system prompt; the same
                {mandi}/{state}/... placeholders are filled from the configured target)
  query_hints   example searches; {year} {month} {month_hi} are filled per window and
                {mandi} {state} {mandi_hi} {state_hi} {commodity} from the configured target
                (config.market_terms), so the hints follow whatever market is set. They are
                starting points the subagent adapts, not a fixed list it must run
  keys          columns identifying one record in data/staged/agent_<name>.parquet
  topic         Tavily search mode: "news" (dated articles) or "general" (web pages)

To add an area: add a schema (schemas.py), an entry here, and a feature method in
features/build.py that reads data/staged/agent_<name>.parquet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from pydantic import BaseModel

from . import schemas as S

# Hindi month names: many Indian mandi reports (Naidunia, Dainik Bhaskar, Krishi Jagran Hindi)
# are in Hindi, and searching with Hindi month names finds them far better than English.
MONTH_HI = ["जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


@dataclass
class ResearchTask:
    """One research area's config (see the module docstring for each field).

    Example:
        TASKS["news_sentiment"].topic   -> 'news'
        TASKS["news_sentiment"].keys    -> ['url']           # what identifies one staged record
        TASKS["news_sentiment"].schema  -> <class 'schemas.NewsScores'>   # the save tool's arg schema
    """

    name: str
    schema: Type[BaseModel]
    record_model: Type[BaseModel]
    description: str
    instructions: str
    query_hints: list[str]
    keys: list[str]
    topic: str = "news"
    include_domains: list[str] = field(default_factory=list)   # optional: restrict searches to these sites


# Rules every researcher follows, whatever the area. The save tools enforce the critical ones
# (verbatim quote present in the document, publish date inside the window) in code.
COMMON = (
    "Only save facts stated in documents you retrieved with your tools. Never use your own knowledge of "
    "prices, policies or events, and never infer or estimate numbers. Every record needs the doc_id it came "
    "from and a short quote copied EXACTLY from that document's text, in its original language (do not "
    "translate the quote). Use the article's publish date for published_at. Dates must be YYYY-MM-DD. "
    "Hindi documents are common: read them carefully (भाव=price, आवक=arrivals, क्विंटल=quintal, बोरी=bag, "
    "तेजी=rise, मंदी=fall), and write descriptions in English."
)

TASKS: dict[str, ResearchTask] = {
    "policy_events": ResearchTask(
        name="policy_events",
        schema=S.PolicyEvents,
        record_model=S.PolicyEvent,
        description="Finds Indian government decisions affecting soybean/soymeal/edible-oil prices "
                    "(MSP, procurement, import duty, tariff values, futures ban, stock limits, export rules).",
        instructions=(
            "Find Indian government policy decisions affecting soybean, soymeal or edible oil prices: MSP, "
            "procurement windows, import duties/tariff values, futures trading suspension or resumption, stock "
            "limits, soymeal export incentives/restrictions, {state} state schemes. One record per distinct "
            "decision, with announcement and effective dates and the numeric level when given."),
        query_hints=[
            "soybean MSP announced {year} kharif",
            "edible oil import duty soybean oil {month} {year}",
            "crude soyabean oil tariff value notification {month} {year}",
            "soymeal export policy India {month} {year}",
            "soybean procurement {state} PSS bhavantar {month} {year}",
            "SEBI NCDEX soybean futures suspension {year}",
        ],
        keys=["event_type", "effective_from", "value"],
    ),
    "industry_stats": ResearchTask(
        name="industry_stats",
        schema=S.IndustryStats,
        record_model=S.IndustryStat,
        description="Finds published statistics: SOPA crop/crush/stocks/meal exports, SEA oil imports, "
                    "state/India soybean area and production, state procurement.",
        instructions=(
            "Find official or industry-body statistics: SOPA crop estimates, monthly crush, stocks, soymeal "
            "exports; SEA edible/soy oil import volumes and port stocks; {state}/India soybean sown area and "
            "production estimates; {state} procurement volumes. Keep the unit exactly as printed and the period "
            "it refers to."),
        query_hints=[
            "SOPA soybean crushing {month} {year}",
            "SOPA soybean crop estimate {year} lakh tonnes",
            "SOPA soymeal export {month} {year}",
            "SEA vegetable oil imports {month} {year} soybean oil",
            "{state} soybean sowing area {year} lakh hectare",
            "{state_hi} सोयाबीन उत्पादन अनुमान {year}",
        ],
        # published_at in the key keeps revised estimates (SOPA Oct vs Dec) as separate rows.
        keys=["metric", "period", "issuer", "published_at"],
    ),
    "product_prices": ResearchTask(
        name="product_prices",
        schema=S.ProductPrices,
        record_model=S.ProductPrice,
        description="Finds dated soymeal (DOC), soy oil, plant-delivery soybean and target-mandi "
                    "prices/arrivals from market reports (mostly Hindi).",
        instructions=(
            "Find dated prices for soymeal (DOC), soy oil (refined {mandi}, crude Kandla), plant-delivery "
            "soybean and {mandi} mandi soybean, plus mandi arrivals when quoted. One record per product per "
            "date. Soymeal is usually Rs/tonne, oil Rs/10kg, seed Rs/quintal. Aim for at least one quote per "
            "product per week of the window."),
        query_hints=[
            "{mandi} soya DOC price {month} {year}",
            "{mandi} soybean plant delivery price {month} {year}",
            "{mandi_hi} सोयाबीन भाव प्लांट डिलीवरी {month_hi} {year}",
            "{mandi_hi} सोया तेल भाव {month_hi} {year}",
            "{mandi_hi} मंडी सोयाबीन आवक {month_hi} {year}",
            "soybean oil refined price {mandi} {month} {year}",
        ],
        keys=["product", "obs_date", "unit"],
    ),
    "news_sentiment": ResearchTask(
        name="news_sentiment",
        schema=S.NewsScores,
        record_model=S.NewsScore,
        description="Scores news articles (Hindi and English) for their implication on the target "
                    "market's soybean prices.",
        instructions=(
            "Find news about soybean prices, arrivals, crop condition, demand and policy relevant to "
            "{mandi}/{state}. Score each relevant article for its implication on {mandi} soybean seed prices "
            "over the coming weeks: sentiment +1 = strongly bullish (prices likely to rise), -1 = strongly "
            "bearish; relevance 0-1. One record per article. Cover the whole window, not just one week."),
        query_hints=[
            "soybean prices mandi {state} {month} {year}",
            "सोयाबीन भाव मंडी {month_hi} {year}",
            "soybean farmers prices India {month} {year}",
            "soymeal soybean oil market India {month} {year}",
        ],
        keys=["url"],
    ),
    "price_forecasts": ResearchTask(
        name="price_forecasts",
        schema=S.PriceForecasts,
        record_model=S.PriceForecast,
        description="Finds explicit soybean/soymeal price outlooks from named forecasters.",
        instructions=(
            "Find explicit forward-looking price views from named forecasters (brokerages, research desks, "
            "SOPA, trade analysts, rating agencies). Include the market, period, range/point and direction. "
            "Ignore generic statements without a forecaster or direction."),
        query_hints=[
            "soybean price forecast India {month} {year}",
            "soybean price outlook {mandi} {month} {year}",
            "सोयाबीन भाव तेजी मंदी अनुमान {month_hi} {year}",
            "soymeal price outlook India {year}",
        ],
        keys=["forecaster", "target_market", "target_period", "published_at"],
    ),
}


def render_hints(task: ResearchTask, start, terms: dict[str, str] | None = None) -> list[str]:
    """Fill a task's query hints for the window starting at `start` (a pandas Timestamp).

    terms : the market names from config.market_terms() ({mandi}, {state}, {mandi_hi},
            {state_hi}, {commodity}); when None the market placeholders are left for the
            caller (tests that only check the date substitution).

    Example:
        render_hints(TASKS["product_prices"], pd.Timestamp("2024-11-01"),
                     {"mandi": "Indore", "mandi_hi": "इंदौर", "state": "Madhya Pradesh",
                      "state_hi": "मध्य प्रदेश", "commodity": "Soyabean"})
          -> ['Indore soya DOC price November 2024', 'इंदौर सोया तेल भाव नवंबर 2024', ...]
    """
    ctx = {"year": start.year, "month": start.strftime("%B"), "month_hi": MONTH_HI[start.month - 1],
           **(terms or {})}
    out = []
    for q in task.query_hints:
        # format_map + _Keep leaves any placeholder we weren't given (e.g. {mandi} when terms
        # is None) untouched, instead of raising KeyError.
        r = q.format_map(_Keep(ctx))
        if r not in out:
            out.append(r)
    return out


class _Keep(dict):
    """A format mapping that returns `{key}` unchanged for keys it doesn't hold."""

    def __missing__(self, key):
        return "{" + key + "}"