"""Record formats the research agent saves (Pydantic models).

Each area's wrapper model `<Name>s(records=[...])` is the argument schema of its save tool
(save_<area> in tools.py), so the agent can only save records of these shapes: no free
text, no invented fields. The single-record models are re-validated one by one on save.

Every record inherits `Evidence`:
  doc_id        the retrieved document it came from ("d3") -> lets us attach the real URL
  quote         verbatim phrase from that document -> checked in code, and lets you audit it
  published_at  article date -> becomes `available_at` (the point-in-time key)
  confidence    0-1 self-rating -> low-confidence records are rejected on save

The Field(description=...) texts are sent to the LLM as part of the tool schema, so they are
written as instructions to the model.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Common provenance fields for every extracted record.

    Example (the fields every record carries):
        {"doc_id": "d3", "quote": "Soybean prices rose Rs 100 in Indore mandi",
         "published_at": "2024-11-06", "confidence": 0.9}
    """

    doc_id: str = Field(description="doc_id of the source document from a search_news result, e.g. 'd3'")
    quote: str = Field(description="short verbatim phrase (<= 25 words) supporting the record")
    published_at: Optional[str] = Field(None, description="article publish date YYYY-MM-DD if stated or known")
    confidence: float = Field(ge=0, le=1)


# --------------------------------------------------------------------------- policy_events
class PolicyEvent(Evidence):
    """One government decision affecting soybean / soymeal / edible-oil prices.

    announced_at vs effective_from matters: the market reacts on announcement, the rule
    applies from the effective date. Both become separate features.

    Example record:
        {"doc_id": "d2", "quote": "government cut the import duty on crude soyoil to 5.5%",
         "confidence": 0.85, "event_type": "import_duty_crude_soyoil", "announced_at": "2024-09-14",
         "effective_from": "2024-09-14", "value": 5.5, "unit": "percent", "direction": "bearish",
         "description": "Crude soyoil import duty cut to 5.5%"}
    """

    # Closed list so features can be built per type; "other" catches anything new.
    event_type: Literal[
        "msp", "procurement_start", "procurement_end", "import_duty_crude_soyoil",
        "import_duty_refined_soyoil", "import_duty_crude_palm", "tariff_value_soyoil",
        "futures_suspension", "futures_resumption", "stock_limit", "export_policy_soymeal",
        "export_incentive", "state_scheme_bhavantar", "gm_soymeal_import", "other"]
    announced_at: Optional[str] = Field(None, description="YYYY-MM-DD the decision was announced")
    effective_from: Optional[str] = Field(None, description="YYYY-MM-DD it took effect")
    effective_to: Optional[str] = None
    value: Optional[float] = Field(None, description="numeric level (e.g. duty %, MSP Rs/qtl, quantity)")
    unit: Optional[str] = None
    direction: Optional[Literal["bullish", "bearish", "neutral"]] = Field(
        None, description="expected effect on domestic soybean seed prices")
    description: str


class PolicyEvents(BaseModel):
    records: list[PolicyEvent] = []


# --------------------------------------------------------------------------- industry_stats
class IndustryStat(Evidence):
    """One published statistic (SOPA, SEA, the state agriculture dept ...).

    `metric` names follow <issuer>_<what>_<unit>: lt = lakh tonnes, lha = lakh hectares.
    The unit is kept exactly as printed; features/build.py converts it to lakh units.

    Example record:
        {"doc_id": "d5", "quote": "SOPA pegged the 2024 soybean crop at 125 lakh tonnes",
         "confidence": 0.8, "metric": "sopa_crop_estimate_lt", "period": "2024-25",
         "value": 125.0, "unit": "lakh tonnes", "issuer": "SOPA"}
    """

    # state_* = the target market's state (settings target.state); national ones stay india_*.
    metric: Literal[
        "sopa_crop_estimate_lt", "sopa_crush_monthly_lt", "sopa_stock_lt", "sopa_meal_export_lt",
        "sea_soyoil_import_lt", "sea_edible_oil_import_lt", "sea_edible_oil_port_stock_lt",
        "state_soy_area_lha", "state_soy_production_lt", "india_soy_area_lha", "india_soy_production_lt",
        "state_procurement_lt", "other"]
    period: str = Field(description="period the figure refers to, e.g. 2026-09, 2025-26, kharif 2026")
    period_end: Optional[str] = Field(None, description="last day of that period YYYY-MM-DD")
    value: float
    unit: str = Field(description="unit as printed, e.g. lakh tonnes, lakh ha")
    issuer: Optional[str] = Field(None, description="SOPA, SEA, state agriculture dept, USDA ...")


class IndustryStats(BaseModel):
    records: list[IndustryStat] = []


# --------------------------------------------------------------------------- product_prices
class ProductPrice(Evidence):
    """One dated price quote for a soy product or the seed itself, from market reports.

    Soymeal (DOC) and soy oil prices feed the crush margin; plant-delivery price shows what
    processors pay vs the mandi. Hindi market reports often quote arrivals in bags (bori).

    Example record:
        {"doc_id": "d1", "quote": "इंदौर सोया डीओसी 42000 रुपये टन", "confidence": 0.9,
         "product": "soymeal_doc_local", "obs_date": "2024-11-06", "price": 42000.0,
         "unit": "rs_per_tonne"}
    """

    # *_local = at/near the target mandi's processing hub; export_fob and crude_kandla are the
    # national benchmarks and stay fixed whatever the target mandi is.
    product: Literal["soymeal_doc_local", "soymeal_doc_export_fob", "soy_oil_refined_local",
                     "soy_oil_crude_kandla", "soy_plant_delivery_local", "soyabean_local_mandi"]
    obs_date: str = Field(description="date the price refers to YYYY-MM-DD")
    price_low: Optional[float] = None
    price_high: Optional[float] = None
    price: float = Field(description="representative price; mid of range if a range")
    unit: Literal["rs_per_tonne", "rs_per_10kg", "rs_per_qtl", "usd_per_tonne"]
    arrivals_bags_or_qtl: Optional[float] = Field(None, description="mandi arrivals if quoted (soyabean only)")
    arrivals_unit: Optional[Literal["bags", "qtl", "tonnes"]] = None


class ProductPrices(BaseModel):
    records: list[ProductPrice] = []


# --------------------------------------------------------------------------- news_sentiment
class NewsScore(Evidence):
    """LLM rating of one news article's implication for the target market's soybean prices.

    relevance weights the article in weekly averages (an article about US soy farmers counts
    less than one about the target mandi's arrivals); topics allow topic-specific sentiment features.

    Example record:
        {"doc_id": "d1", "quote": "Soybean prices rose Rs 100 in Indore mandi", "confidence": 0.9,
         "published_at": "2024-11-06", "headline": "Soybean prices rise in Indore", "relevance": 0.9,
         "sentiment": 0.6, "topics": ["supply"], "horizon": "weeks"}
    """

    headline: str
    relevance: float = Field(ge=0, le=1, description="relevance to Indian soybean seed prices")
    sentiment: float = Field(ge=-1, le=1, description="-1 very bearish ... +1 very bullish for the target market's soybean price")
    topics: list[Literal["supply", "demand", "policy", "weather", "global", "crush_margin",
                         "farmer_holding", "quality", "logistics", "other"]] = []
    horizon: Literal["days", "weeks", "months", "season"] = "weeks"


class NewsScores(BaseModel):
    records: list[NewsScore] = []


# --------------------------------------------------------------------------- price_forecasts
class PriceForecast(Evidence):
    """An explicit forward-looking price view by a named forecaster (brokerage, SOPA, analyst).

    Only direction is required, because many reports say "prices likely to firm up" without
    numbers; ranges / % change are captured when given.

    Example record:
        {"doc_id": "d4", "quote": "SOPA expects soybean to trade firm towards Rs 5000 by December",
         "confidence": 0.7, "forecaster": "SOPA", "target_market": "Indore mandi",
         "target_period": "Dec 2024", "price_point": 5000.0, "unit": "rs_per_qtl", "direction": "up"}
    """

    forecaster: str = Field(description="organisation or analyst making the forecast")
    target_market: str = Field(description="e.g. the target mandi, state mandis, India plant delivery, CBOT")
    target_period: str = Field(description="when the forecast applies, e.g. Nov 2026, next 2 weeks")
    target_date: Optional[str] = Field(None, description="end date of that period YYYY-MM-DD")
    price_low: Optional[float] = None
    price_high: Optional[float] = None
    price_point: Optional[float] = None
    unit: Optional[str] = None
    direction: Literal["up", "down", "sideways"]
    expected_change_pct: Optional[float] = None


class PriceForecasts(BaseModel):
    records: list[PriceForecast] = []