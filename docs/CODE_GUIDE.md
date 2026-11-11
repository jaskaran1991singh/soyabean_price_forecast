# Code guide — how to read this project

A walkthrough for a developer new to `soyabean_price_forecast`. Read this top to bottom once,
then use it as a map. It tells you **what the project does**, **which file to open in what order**,
and **what lives in each file**.

---

## 1. What the project does (in one paragraph)

It builds the **training & test data for a model that forecasts the weekly soybean price at the
Indore mandi** (Madhya Pradesh, India). It pulls raw data from several public sources, cleans each
into a tidy table, optionally enriches it with an LLM research agent, and assembles **one row per
week** of point-in-time features plus the forward-looking price labels. The output is a few files
under `data/features/` that you can hand to a forecasting model (tree/linear, or a Time-LLM/TimesFM
style model). It does **not** train the model — it produces the dataset.

Indore is just the **default target**, not baked in: the market, its state/district, the peer
mandis and the weather catchment all come from `config/settings.yaml` (`target` + `peer_markets` +
`weather.points`). Point those at another mandi and the collectors, the research agent's prompts and
searches, and the output file names all follow — see §6.

---

## 2. The mental model: a 3-stage pipeline

Everything in the code is one of three stages. Hold this picture in your head and every file finds
its place:

```
      STAGE 1: collect            STAGE 2: research         STAGE 3: build
   collectors/*.py  ───┐        agent/*.py  ───┐
   raw public APIs     │        Tavily + LLM    │
        │              ▼              │          ▼
        └────►  data/staged/*.parquet  ◄─────────┘   ───►  features/build.py  ───►  data/features/
               (one cleaned table per source,                (one row per WEEK,       training_weekly.*
                each row tagged obs_date + available_at)       point-in-time safe)     timellm_*.csv
```

- **Stage 1 — collect** (`collectors/`): fetch structured sources (mandi prices, weather, global
  markets, USDA balance sheets), clean them, write `data/staged/<source>.parquet`.
- **Stage 2 — research** (`agent/`): an LLM agent searches the web (Tavily) for things that have no
  clean API — policy decisions, industry stats, product prices, news sentiment, price outlooks —
  and writes more staged tables. **Optional**; the dataset is usable without it.
- **Stage 3 — build** (`features/build.py`): read every staged table and compute the weekly feature
  table + labels, writing the final files to `data/features/`.

The glue is the CLI, `src/soyfc/__main__.py` (`python -m soyfc <command>`), which just runs these
stages.

---

## 3. The two ideas you must understand before reading features

These appear everywhere; learn them first or the feature code won't make sense.

1. **Two dates on every staged row** (defined in `store.py`):
   - `obs_date` — the day the value *describes* (e.g. a mandi trading day).
   - `available_at` — the earliest day the value *could be known* (e.g. trading day + 1, or an
     article's publish date).

2. **The week grid and the cutoff** (in `features/build.py`):
   - One row per week, indexed by `week_end` (Friday).
   - `cutoff = week_end + forecast_lag_days` (Saturday) — the moment the forecast is issued.
   - A feature for week *W* may only use rows with `available_at <= cutoff`. That single rule is
     what keeps the future out of the training data (no look-ahead / no leakage).

If you remember nothing else: **features look backward from the cutoff; labels look forward.**

---

## 4. Recommended reading order

Open the files in this order. Times are rough "read + understand" estimates.

| # | File | Why read it here | ~time |
|---|------|------------------|-------|
| 1 | `src/soyfc/__main__.py` | The table of contents — the CLI commands and what each stage calls. | 10 min |
| 2 | `src/soyfc/config.py` | How settings + secrets load; the `Settings` object used everywhere. | 10 min |
| 3 | `config/settings.yaml` | The knobs: target mandi, horizons, sources, agent config. Read alongside `config.py`. | 10 min |
| 4 | `src/soyfc/store.py` | The I/O layer and the **two-date rule**. Short and foundational. | 15 min |
| 5 | `src/soyfc/collectors/http.py` + **one** collector (`weather.py`) | The shape every collector shares: fetch → clean → `available_at` → `store.upsert`. | 20 min |
| 6 | `src/soyfc/features/build.py` | **The heart.** The week grid, `asof`, and one function per feature group. | 60 min |
| 7 | `src/soyfc/agent/` (tasks → schemas → tools → agent → runner) | The research agent, if you'll touch it. | 45 min |
| 8 | `src/soyfc/**/test_*.py` | Executable specs — especially `features/test_features.py::test_no_lookahead`. | 15 min |

Supporting docs in this folder:
- `docs/DATA_DICTIONARY.md` — every output column, with meaning, unit, source and availability rule.
- `run_pipeline.ipynb` (repo root) — a runnable notebook that executes each stage so you can watch it.

---

## 5. What's in each file

### Entry point
- **`src/soyfc/__main__.py`** — the CLI. `python -m soyfc collect|agent|build|status|run-all|probe-*`.
  Each command loads `Settings`, then calls into a stage. `run-all` = collect all + agent `--update`
  + build (the weekly job). Start here to see how the pieces connect.

### Foundations (used by every stage)
- **`src/soyfc/config.py`** — `load_settings()` reads `config/settings.yaml` and `.env`; returns a
  `Settings` object (dict-style access `s["target"]` plus shortcuts `s.start_date`, `s.week_rule`,
  `s.path(...)`). `env("KEY")` is the only place API keys are read.
- **`src/soyfc/store.py`** — storage as plain functions (no object): `read(s, table)`,
  `upsert(s, table, df, keys)`, `write(...)`, and the `cache_*` helpers for raw API responses.
  Defines the `obs_date` / `available_at` convention. Collectors, the agent and the builder all
  `from .. import store` and call `store.read(s, ...)`.
- **`config/settings.yaml`** — all configuration: the target market (Indore soyabean), forecast
  horizons, peer mandis, each data source's settings, weather points/season windows, USDA/FRED
  details, and the agent config. Non-secret; secrets go in `.env` (see `.env.example`).

### Stage 1 — collectors (`src/soyfc/collectors/`)
Every collector exposes `collect(s) -> DataFrame` and writes one staged table. Same recipe: fetch →
`clean`/normalise → set `available_at` → `store.upsert`.
- **`http.py`** — a shared `get()` with retries/back-off and a `PermanentHTTPError` for 4xx. Read
  this before any collector.
- **`mandi.py`** — the forecast target: Indore + peer mandi daily prices/arrivals, from three sources
  tried in order (mandi-api, data.gov.in, manual Agmarknet CSV). The biggest collector; has the
  header-normalisation and price-cleaning helpers. → `mandi_daily`.
- **`weather.py`** — Open-Meteo ERA5 daily weather for the Indore catchment (keyless). → `weather_daily`.
- **`markets.py`** — yfinance futures/FX + FRED monthly commodity prices. → `markets_daily`.
- **`usda_psd.py`** — USDA PSD global soybean balance sheets (with a 15-month availability lag to
  avoid look-ahead). → `psd_annual`.

### Stage 3 — feature builder (`src/soyfc/features/build.py`)  ← the heart
Plain functions. `build(s)` makes the week grid and calls one function per feature group, each
returning a week-indexed DataFrame; then it joins them and adds the `target_*` labels. Key pieces:
- `week_grid(s)` — the weekly skeleton with `cutoff`.
- `asof(grid, df, cols)` — "latest value known by each week's cutoff" (levels: prices, stocks, MSP).
- `first_usable_week(...)` — which week an event/flow falls into (news, policy, product quotes).
- Feature groups: `target`, `peers`, `markets`, `weather`, `psd`, `policy`, `industry`, `products`,
  `sentiment`, `forecasts`, `calendar`.
- `write_outputs(...)` — writes `training_weekly.parquet/.csv`, `timellm_<market>_<commodity>.csv`
  (named from settings, e.g. `timellm_indore_soyabean.csv`), `data_dictionary.csv`, `coverage_report.md`.
  See `docs/DATA_DICTIONARY.md` for what each output column means.

### Stage 2 — research agent (`src/soyfc/agent/`)
A standard LangChain tool-calling agent: one researcher per area, per time window, looped over
windows in plain Python. Read in this order:
- **`tasks.py`** — the 5 research areas as data (`TASKS`): what each looks for, example queries,
  staging keys.
- **`schemas.py`** — the exact record shape the LLM must return for each area (Pydantic models).
- **`tools.py`** — the `search_news` / `read_doc` / `save_<area>` tools and, crucially, the
  **guards in code**: window-pinning, search budget, verbatim-quote check, `available_at`. The LLM
  decides *what* to search; this file decides what's *allowed*.
- **`agent.py`** — builds one `create_agent(...)` for an area+window and runs the loop.
- **`runner.py`** — the window loop: plan windows, run each not-yet-done one, stage results, track
  progress in `data/agent_progress.parquet` (so re-runs skip finished work, retry failed windows).

### Tests (executable specs)
- **`src/soyfc/features/test_features.py`** — `test_no_lookahead` is the one to read: it proves a
  week's features don't change when data knowable only after the cutoff is removed.
- **`src/soyfc/agent/test_agent.py`** — the tool guards, the runner loop, and a full agent run driven
  by a scripted fake model (no network).

---

## 6. Where do I look if I want to…

| I want to… | Open |
|---|---|
| …change the target mandi, horizons, or date range | `config/settings.yaml` (each market gets its own `data/<market>_<commodity>/` folder) |
| …add/modify a cleaned source | the matching `collectors/*.py` (+ `store.upsert` key) |
| …add or change a feature | the group function in `features/build.py`, then check `DATA_DICTIONARY.md` |
| …understand an output column | `docs/DATA_DICTIONARY.md` |
| …add a new agent research area | `agent/tasks.py` + a model in `agent/schemas.py` + a feature fn in `build.py` |
| …change what the agent is allowed to save | `agent/tools.py` |
| …see the whole thing run | `run_pipeline.ipynb`, or the CLI (below) |
| …verify I didn't introduce look-ahead | `features/test_features.py` |

---

## 7. Run it yourself (fastest way to understand)

Set up an environment and install `requirements.txt`, then:

```
python -m soyfc status                 # what's staged so far
python -m soyfc collect weather        # run one keyless collector
python -m soyfc build                  # build the weekly training files
python -m soyfc agent --dry-run        # show the agent's plan (no API calls, no cost)
python -m soyfc run-all                # the full weekly job (collect + agent update + build)
```

The agent needs `TAVILY_API_KEY` and `GOOGLE_API_KEY` in `.env` (see `.env.example`). The
collectors and `build` do not. `run_pipeline.ipynb` walks the same steps cell by cell and includes a
synthetic-mandi fallback so `build` produces output even before you wire up live mandi prices.

Then open `data/features/`:
- `training_weekly.parquet` / `.csv` — every feature + every `target_*` label.
- `timellm_<market>_<commodity>.csv` (e.g. `timellm_indore_soyabean.csv`) — date + numeric
  covariates + target `y` (no labels), for Time-LLM/TimesFM.
- `data_dictionary.csv` — the columns actually produced, with % filled.
- `coverage_report.md` — how filled each feature group is per year.

---

## 8. Conventions worth knowing

- **Low OOP by design.** `store` and the feature builder are modules of functions, not classes; the
  only classes are small data containers (`Settings`, `Registry`, `ResearchTask`, the Pydantic
  schemas, the agent's per-run `Window`).
- **Every function has a docstring with an `Example:`** showing representative input → output. Read
  the docstring before the body.
- **Missing sources degrade gracefully.** If a source wasn't collected, `store.read` returns an empty
  DataFrame and that feature group is simply absent — `build` still runs.
- **Secrets only in `.env`**, never in `settings.yaml` or code; read exclusively via `config.env()`.
- **Data is partitioned per market.** Everything the pipeline generates lives under
  `data/<market>_<commodity>/` (e.g. `data/indore_soyabean/{raw,staged,features}` + `agent_progress.parquet`),
  resolved by `Settings.work_dir` from `target` in settings. Switching `target.market` points the
  pipeline at a new folder, so one mandi's staged tables can never overwrite another's or corrupt its
  features. (Paths written as `data/staged/...` in this guide are really `data/<market>_<commodity>/staged/...`.)
  The one exception is the manual Agmarknet input dir (`mandi_sources.csv.import_dir`), which stays
  shared — it's your input, and the mandi cleaner filters it by market.
```
