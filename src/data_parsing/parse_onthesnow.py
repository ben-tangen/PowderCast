#!/usr/bin/env python3
"""
parse_onthesnow.py

Find a resort on OnTheSnow and scrape its season-by-season snowfall.

Why discovery instead of guessing URLs: OnTheSnow slugs don't follow the resort's
name. Examples from the Colorado listing page:
    "Copper Mountain" -> /colorado/copper-mountain-resort/
    "Purgatory"       -> /colorado/durango-mountain-resort/
    "Loveland"        -> /colorado/loveland/
So we read each region's listing page (/<region>/skireport), collect every
/<region>/<slug>/skireport link, and match resort names against those.

Pipeline:
    Fetcher                    polite, cached HTTP (1.5 s delay, 404s cached too)
    discover_region()          listing page -> [Listing(region, slug, names)]
    match_resort()             exact normalised match, then conservative fuzzy match
    resolve_resort()           override -> listing match -> guessed URLs
    parse_historical_snowfall  HTML -> DataFrame[season_start_year, season_snowfall_in]
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("onthesnow")

BASE = "https://www.onthesnow.com"
CACHE_DIR = Path("data/raw/onthesnow")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ski-dataset-builder/1.0; research use)",
    "Accept-Language": "en-US,en;q=0.9",
}


# --------------------------------------------------------------------------- #
# 1. Polite, cached HTTP
# --------------------------------------------------------------------------- #
class FetchBlocked(RuntimeError):
    """Raised when several URLs in a row fail: offline, or the site is blocking scripts."""


class Fetcher:
    """GET pages with an on-disk cache so re-runs never re-hit the site.

    Successful pages are stored as .html, 404s as empty .missing marker files.
    Pass use_cache=False to force fresh downloads.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, delay: float = 1.5,
                 use_cache: bool = True, session=None, timeout: int = 30,
                 max_consecutive_failures: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay, self.use_cache, self.timeout = delay, use_cache, timeout
        self.session = session or requests.Session()
        self._last_request = 0.0
        self.network_requests = 0
        self.max_failures = max_consecutive_failures
        self.consecutive_failures = 0

    def _path(self, url: str) -> Path:
        key = re.sub(r"[^a-z0-9]+", "_", url.lower().replace(BASE, "")).strip("_") or "home"
        return self.cache_dir / f"{key}.html"

    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, url: str) -> str | None:
        """Return page HTML, or None if the page doesn't exist / can't be fetched."""
        path = self._path(url)
        missing = path.with_suffix(".missing")
        if self.use_cache:
            if path.exists():
                return path.read_text(encoding="utf-8")
            if missing.exists():
                return None

        for attempt in range(3):
            self._throttle()
            self.network_requests += 1
            try:
                resp = self.session.get(url, timeout=self.timeout, headers=HEADERS)
            except requests.RequestException as exc:
                log.warning("Request failed (%s): %s", exc.__class__.__name__, url)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                text = resp.content.decode("utf-8", errors="replace")
                path.write_text(text, encoding="utf-8")
                self.consecutive_failures = 0
                return text
            if resp.status_code in (404, 410):
                missing.write_text("")
                self.consecutive_failures = 0
                return None
            log.warning("HTTP %s for %s (attempt %d/3)", resp.status_code, url, attempt + 1)
            time.sleep(2 ** attempt)

        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_failures:
            raise FetchBlocked(
                f"{self.consecutive_failures} URLs in a row failed (last: {url}). You may be "
                "offline, or OnTheSnow may be blocking scripted requests. Everything fetched "
                "so far is cached, so just re-run once the problem is fixed.")
        return None


# --------------------------------------------------------------------------- #
# 2. Name normalisation + slug helpers
# --------------------------------------------------------------------------- #
_STOP = {"ski", "resort", "resorts", "area", "mountain", "mountains", "the", "at", "and", "of"}
_ALIASES = {"mt": "mount", "mtn": "mountain", "st": "saint"}


def norm_name(text: str) -> str:
    """'Crested Butte Ski Resort' and 'Crested Butte Mountain Resort' -> 'crested butte'."""
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s.replace("&", " and "))
    tokens = [_ALIASES.get(t, t) for t in s.split()]
    kept = [t for t in tokens if t not in _STOP]
    return " ".join(kept or tokens)


def slugify(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def slug_variants(name: str) -> list[str]:
    """A few plausible slugs for a resort name, most specific first (used only as a fallback)."""
    base = slugify(name)
    variants = [base]
    for suffix in ("-ski-resort", "-mountain-resort", "-resort", "-ski-area", "-mountain"):
        if base.endswith(suffix):
            variants.append(base[: -len(suffix)])
    variants.append(slugify(norm_name(name)))
    return list(dict.fromkeys(v for v in variants if v))[:4]


# --------------------------------------------------------------------------- #
# 3. Discover resorts from a region's listing page
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Listing:
    region: str
    slug: str
    names: tuple[str, ...]          # slug-derived name + any clean anchor texts

    @property
    def path(self) -> str:
        return f"{self.region}/{self.slug}"


_LINK_RE = re.compile(
    r'href="(?:https?://(?:www\.)?onthesnow\.com)?/([a-z0-9-]+)/([a-z0-9-]+)/skireport(?:\.html)?'
    r'(?:[?#][^"]*)?"[^>]*>(.*?)</a>', re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def parse_listing(html: str) -> list[Listing]:
    """All /<region>/<slug>/skireport links on a page, with their display names."""
    names: dict[tuple[str, str], set[str]] = {}
    for m in _LINK_RE.finditer(html):
        region, slug, inner = m.group(1).lower(), m.group(2).lower(), m.group(3)
        bucket = names.setdefault((region, slug), {slug.replace("-", " ")})
        text = re.sub(r"\s+", " ", _TAG_RE.sub(" ", inner)).strip()
        # Anchor text sometimes has snow totals glued on ('Alta 4"'): keep only clean names.
        if 2 <= len(text) <= 60 and not re.search(r'[\d"%/]', text):
            bucket.add(text)
    return [Listing(r, s, tuple(sorted(n))) for (r, s), n in names.items()]


def discover_region(fetcher: Fetcher, region: str) -> list[Listing]:
    html = fetcher.get(f"{BASE}/{region}/skireport")
    if not html:
        log.info("No listing page for region %r", region)
        return []
    found = parse_listing(html)
    log.info("Region %-18s %3d resorts listed", region, len(found))
    return found


# --------------------------------------------------------------------------- #
# 4. Match a resort name to a listing
# --------------------------------------------------------------------------- #
@dataclass
class Match:
    path: str | None                # e.g. 'utah/alta-ski-area'
    method: str                     # override | exact | fuzzy | guess | none
    score: float = 0.0
    matched_name: str = ""
    best_guess: str = ""            # closest listing when nothing was accepted (for the report)


FUZZY_CUTOFF = 0.90   # similarity needed for a fuzzy match
FUZZY_MARGIN = 0.04   # best must beat the runner-up by this much (avoids Copper vs Cooper)


def match_resort(name: str, listings: list[Listing]) -> Match:
    key = norm_name(name)
    normed = [(lst, {norm_name(n) for n in lst.names}) for lst in listings]

    exact = [lst for lst, ns in normed if key in ns]
    if exact:
        # Prefer the listing whose own slug equals the name, if several tie.
        exact.sort(key=lambda lst: norm_name(lst.slug.replace("-", " ")) != key)
        return Match(exact[0].path, "exact", 1.0, exact[0].names[0])

    scored = sorted(((max(SequenceMatcher(None, key, n).ratio() for n in ns), lst)
                     for lst, ns in normed), key=lambda t: -t[0])
    if scored:
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= FUZZY_CUTOFF and best_score - runner_up >= FUZZY_MARGIN:
            return Match(best.path, "fuzzy", round(best_score, 3), best.names[0])
        return Match(None, "none", round(best_score, 3), "", f"{best.path} ({best_score:.2f})")
    return Match(None, "none")


def resolve_resort(name: str, regions: list[str], fetcher: Fetcher,
                   listing_cache: dict[str, list[Listing]],
                   overrides: dict[str, str] | None = None,
                   allow_guess: bool = True) -> Match:
    """override -> listing match -> guessed URLs (each guess is validated by parsing)."""
    if overrides and name in overrides:
        return Match(overrides[name].strip("/"), "override", 1.0)

    pool: list[Listing] = []
    for region in regions:
        if region not in listing_cache:
            listing_cache[region] = discover_region(fetcher, region)
        pool.extend(listing_cache[region])
    pool = list({lst.path: lst for lst in pool}.values())

    match = match_resort(name, pool)
    if match.path or not allow_guess or not regions:
        return match

    for slug in slug_variants(name)[:3]:
        path = f"{regions[0]}/{slug}"
        if get_snowfall(fetcher, path) is not None:
            return Match(path, "guess", 0.5, slug, match.best_guess)
    return match


# --------------------------------------------------------------------------- #
# 5. Parse the historical-snowfall page
# --------------------------------------------------------------------------- #
# '2012 — 2013', '2012 - 2013', '2025 — To Date'
_SEASON_RE = re.compile(r"^\s*(\d{4})\s*[\u2014\u2013-]+\s*(\d{4}|to\s+date)\s*$", re.I)
_SNOW_RE = re.compile(r"^\s*([\d,]*\.?\d+)\s*(\"|\u201d|\u2033|in\b|cm\b|mm\b)?", re.I)


def parse_snow_value(text) -> float:
    """'366"' -> 366.0 ; '930cm' -> 366.1 ; no unit -> assumed inches ; junk -> NaN."""
    if pd.isna(text):
        return np.nan
    m = _SNOW_RE.match(str(text))
    if not m:
        return np.nan
    value = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or '"').lower()
    if unit == "cm":
        return round(value / 2.54, 1)
    if unit == "mm":
        return round(value / 25.4, 1)
    return value


def parse_historical_snowfall(html: str) -> pd.DataFrame | None:
    """
    Return complete seasons only: columns season_start_year, season_snowfall_in.
    'To Date' (in-progress) and 'Average' rows are dropped. None if no annual table found.
    """
    try:
        # flavor pinned: otherwise pandas falls back to html5lib (often not installed)
        # when a page has no tables, and the ImportError would kill the whole run.
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        return None

    for table in tables:
        if table.shape[1] < 2:
            continue
        labels = table.iloc[:, 0].astype(str)
        is_season = labels.map(lambda s: bool(_SEASON_RE.match(s)))
        if is_season.sum() < 2:
            continue                                   # monthly table or something else

        headers = [str(c).lower() for c in table.columns]
        snow_idx = next((i for i, h in enumerate(headers) if "total snowfall" in h), 1)

        rows = []
        for label, raw_value in zip(labels[is_season], table.iloc[:, snow_idx][is_season]):
            m = _SEASON_RE.match(label)
            start, end = int(m.group(1)), m.group(2).lower()
            if end.startswith("to"):                   # in-progress season
                continue
            if int(end) != start + 1:
                log.warning("Odd season label %r - skipped", label)
                continue
            value = parse_snow_value(raw_value)
            if np.isnan(value):
                continue
            rows.append({"season_start_year": start, "season_snowfall_in": value})
        if rows:
            return (pd.DataFrame(rows).drop_duplicates("season_start_year", keep="last")
                    .sort_values("season_start_year").reset_index(drop=True))
    return None


def get_snowfall(fetcher: Fetcher, path: str) -> pd.DataFrame | None:
    """Fetch + parse /<path>/historical-snowfall. None if the page or table is missing."""
    html = fetcher.get(f"{BASE}/{path}/historical-snowfall")
    if not html:
        return None
    return parse_historical_snowfall(html)