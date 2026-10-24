#!/usr/bin/env python3
"""
get_resort_latitude_longitude.py

Look up latitude/longitude for ski resorts with OpenStreetMap's Nominatim API
(free, no key). OSM maps most ski areas directly (landuse=winter_sports), so the
result is usually the resort itself rather than the nearest town.

Safeguards:
  * results are restricted to the resort's country, and
  * we record whether the returned state/province matches the CSV's
    (coord_state_match). A False there deserves a manual look.
Nominatim's usage policy: max 1 request/second and an identifying User-Agent, so
pass your contact email (see build_ski_dataset.py --contact-email). Every lookup is
cached in data/raw/geocode_cache.json, so re-runs make no requests.

Hand-fix any wrong result by adding a row to data/raw/coordinate_overrides.csv
(columns: resort,latitude,longitude); overrides always win.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from pathlib import Path

import requests

log = logging.getLogger("geocode")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CACHE_PATH = Path("data/raw/geocode_cache.json")

COUNTRY_CODES = {
    "United States": "us", "Canada": "ca", "Switzerland": "ch", "Australia": "au",
    "Japan": "jp", "France": "fr", "Italy": "it", "Austria": "at",
    "New Zealand": "nz", "Andorra": "ad", "Chile": "cl",
}
_SUFFIXES = re.compile(r"\b(ski resort|mountain resort|ski area|resort)\b", re.I)


def _plain(text: str) -> str:
    s = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def core_name(name: str) -> str:
    return re.sub(r"\s+", " ", _SUFFIXES.sub("", name)).strip(" -/") or name


class Geocoder:
    def __init__(self, contact: str = "", cache_path: Path = CACHE_PATH,
                 delay: float = 1.1, session=None, use_cache: bool = True, timeout: int = 30):
        ua = "ski-dataset-builder/1.0 (research)" + (f" contact: {contact}" if contact else "")
        self.headers = {"User-Agent": ua}
        self.cache_path = Path(cache_path)
        self.delay, self.timeout = delay, timeout
        self.session = session or requests.Session()
        self.use_cache = use_cache
        self._last = 0.0
        self.network_requests = 0
        self.cache: dict[str, list] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Ignoring unreadable geocode cache %s", self.cache_path)

    # ------------------------------------------------------------------ #
    def _search(self, query: str, country_code: str | None) -> list[dict]:
        key = f"{country_code or '*'}|{query}"
        if self.use_cache and key in self.cache:
            return self.cache[key]

        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        params = {"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1}
        if country_code:
            params["countrycodes"] = country_code
        self.network_requests += 1
        try:
            resp = self.session.get(NOMINATIM_URL, params=params,
                                    headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            results = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Geocode request failed for %r (%s)", query, exc.__class__.__name__)
            return []                       # not cached, so a re-run retries
        self.cache[key] = results
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        return results

    @staticmethod
    def _is_skiish(result: dict) -> bool:
        blob = f"{result.get('type', '')} {result.get('display_name', '')}".lower()
        return any(w in blob for w in ("winter_sports", "ski", "piste"))

    @staticmethod
    def _state_matches(result: dict, state: str) -> bool | None:
        """True/False if we can compare states, None if the CSV state isn't comparable."""
        wanted = [_plain(s) for s in re.split(r"[/,]", state or "") if s.strip()]
        addr = result.get("address", {})
        got = _plain(" ".join(str(addr.get(k, "")) for k in
                              ("state", "province", "region", "state_district", "county")))
        if not wanted or not got:
            return None
        return any(w in got or got in w for w in wanted)

    # ------------------------------------------------------------------ #
    def geocode(self, name: str, state: str, country: str) -> dict | None:
        """Best (latitude, longitude, ...) for a resort, or None if nothing plausible."""
        cc = COUNTRY_CODES.get(country)
        core = core_name(name)
        queries = [f"{name}, {state}, {country}",
                   f"{core} ski area, {state}, {country}",
                   f"{core}, {state}, {country}"]
        check_state = country in ("United States", "Canada")

        fallback = None
        for query in dict.fromkeys(queries):
            results = [r for r in self._search(query, cc)
                       if not cc or r.get("address", {}).get("country_code", cc) == cc]
            results.sort(key=lambda r: not self._is_skiish(r))      # ski-mapped results first
            for r in results:
                state_ok = self._state_matches(r, state) if check_state else None
                hit = {"latitude": round(float(r["lat"]), 4),
                       "longitude": round(float(r["lon"]), 4),
                       "coord_state_match": state_ok,
                       "coord_display_name": r.get("display_name", ""),
                       "coord_query": query}
                if state_ok in (True, None):
                    return hit
                fallback = fallback or hit
        if fallback:
            log.warning("%s: only a state-mismatched geocode found (%s) - check it",
                        name, fallback["coord_display_name"][:70])
        return fallback
