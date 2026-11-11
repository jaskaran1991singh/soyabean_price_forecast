"""Small HTTP helper shared by all collectors: GET with retries and back-off.

Public data portals (data.gov.in, Open-Meteo, FRED, USDA) occasionally rate-limit or time
out, so every download goes through get(), which retries temporary failures and fails fast
on permanent ones (wrong URL, bad key).
"""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

# Identify the pipeline politely; some portals reject requests without a User-Agent.
UA = {"User-Agent": "soyfc/0.1 (research pipeline)"}

# Status codes worth retrying: 429 = rate-limited, 5xx = server-side trouble.
RETRY_STATUS = {429, 500, 502, 503, 504}


class PermanentHTTPError(RuntimeError):
    """4xx error (other than 429): retrying won't help, e.g. 401 bad key or 404 wrong URL."""


def get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 60, **kw) -> requests.Response:
    """GET a URL and return the response, retrying temporary failures.

    url     : full URL
    params  : query-string parameters (requests URL-encodes them)
    retries : total attempts; waits 1s, 2s, 4s ... between attempts (exponential back-off)
    timeout : seconds to wait for the server per attempt
    **kw    : passed through to requests.get (e.g. headers={...})
    Raises PermanentHTTPError immediately for 4xx, RuntimeError after all retries fail.

    Example:
        r = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": "PSOILUSDM"})
        r.status_code   -> 200
        r.text[:40]     -> 'observation_date,PSOILUSDM\\n2015-01-01,...'
        get("https://api.data.gov.in/resource/bad-id")  -> raises PermanentHTTPError (404, no retry)
    """
    headers = {**UA, **kw.pop("headers", {})}  # merge caller headers once, before the loop
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout, **kw)
        except requests.RequestException as e:      # network error / timeout -> retry
            last = e
        else:
            if r.status_code in RETRY_STATUS:       # temporary server problem -> retry
                last = requests.HTTPError(f"{r.status_code} from {url}")
            elif 400 <= r.status_code < 500:         # our fault -> stop now, show the reason
                raise PermanentHTTPError(f"{r.status_code} from {url}: {r.text[:300]}")
            else:
                r.raise_for_status()
                return r
        wait = 2 ** i
        log.warning("GET %s failed (%s); retry %d/%d in %ss", url, last, i + 1, retries, wait)
        time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")