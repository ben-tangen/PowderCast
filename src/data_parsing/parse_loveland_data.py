#!/usr/bin/env python3
"""
loveland_history.py
 
Build a one-row-per-season table for Loveland Ski Area from two official pages:
 
    https://skiloveland.com/openingclosingdates/   opening / closing dates (1979-80 on)
    https://skiloveland.com/snowfallhistory/       season snowfall totals  (1968/69 on)
 
Output columns follow your target table:
    resort, season, opening_date, closing_date, season_length, season_snowfall_in
plus data-quality flags (see the FLAGS section in the README block below).
 
Usage
-----
    python loveland_history.py                  # download both pages, write CSVs
    python loveland_history.py --use-cache      # re-parse data/raw/ copies, no download
    python loveland_history.py --opening-html a.html --snowfall-html b.html
 
From Python:
    from loveland_history import load_loveland_history
    seasons, opening, snowfall = load_loveland_history()
 
FLAGS
-----
    covid_shutdown            2019-20 closed early (Mar 14, 2020) for COVID, so that
                              season's closing date / length isn't weather-driven.
    date_year_corrected       the page's year disagreed with the season label (e.g.
                              2025-26 opening typed as 11/8/26); we trust the label.
    snowfall_repeated_block   snowfall value belongs to a run that exactly repeats an
                              earlier run (see detect_repeated_blocks) -> likely bad data.
    length_ok / snowfall_ok   convenience booleans: True when the value is present and
                              not covid_shutdown (length) / not snowfall_repeated_block
                              (snowfall). date_year_corrected is informational only: the
                              fix is unambiguous, so those rows stay usable.
 
Requires: pandas, requests, lxml
"""
from __future__ import annotations
 
import argparse
import logging
import re
from io import StringIO
from pathlib import Path
 
import numpy as np
import pandas as pd
import requests
 
log = logging.getLogger("loveland")
 
RESORT = "Loveland"
OPENING_URL = "https://skiloveland.com/openingclosingdates/"
SNOWFALL_URL = "https://skiloveland.com/snowfallhistory/"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
 
 
# --------------------------------------------------------------------------- #
# 1. Download + find the right table
# --------------------------------------------------------------------------- #
def fetch_html(url: str, cache_path: Path, use_cache: bool = False, timeout: int = 30) -> str:
    """Download a page (or reuse the saved copy) and keep the raw HTML on disk."""
    cache_path = Path(cache_path)
    if use_cache and cache_path.exists():
        log.info("Using cached HTML: %s", cache_path)
        return cache_path.read_text(encoding="utf-8")
 
    log.info("Downloading %s", url)
    resp = requests.get(
        url, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (research script; snowfall-history)"},
    )
    resp.raise_for_status()
    html = resp.content.decode("utf-8", errors="replace")
 
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    log.info("Saved raw HTML to %s", cache_path)
    return html
 
 
def find_table(html: str, *required_headers: str) -> pd.DataFrame:
    """Return the first table whose headers contain every required (lowercase) substring."""
    for table in pd.read_html(StringIO(html)):
        headers = [str(c).strip().lower() for c in table.columns]
        if all(any(req in h for h in headers) for req in required_headers):
            return table
    raise ValueError(f"No table with headers containing {required_headers}. "
                     "The page layout may have changed.")
 
 
# --------------------------------------------------------------------------- #
# 2. Cell parsers
# --------------------------------------------------------------------------- #
_NOTE_RE = re.compile(r"^(.*?)\s*\(([^)]*)\)\s*$")
_SEASON_RE = re.compile(r"^\s*(\d{4})\s*[-/\u2013]\s*(\d{2,4})\s*$")
_MDY_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\s*$")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?|\.\d+")
 
 
def split_note(value) -> tuple[str, str]:
    """'10/07/09 (Earliest Opening)' -> ('10/07/09', 'Earliest Opening')."""
    if pd.isna(value):
        return "", ""
    text = str(value).strip()
    m = _NOTE_RE.match(text)
    return (m.group(1).strip(), m.group(2).strip()) if m else (text, "")
 
 
def parse_season(label) -> int | None:
    """'1979-80' or '1968/69' -> 1979 / 1968 (the start year). None if not a season label."""
    m = _SEASON_RE.match(str(label))
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2)) % 100
    if end != (start + 1) % 100:
        log.warning("Season label %r is not consecutive years", label)
    return start
 
 
def parse_number(text: str) -> float:
    m = _NUM_RE.search(text or "")
    return float(m.group()) if m else np.nan
 
 
def parse_mdy(text: str, expected_year: int) -> tuple[pd.Timestamp, bool]:
    """
    Parse 'MM/DD/YY'. The year is taken from the season label (opening = start year,
    closing = start year + 1) rather than guessed from the 2-digit year; if the page's
    year disagrees we use the label's and report that a correction was made.
    """
    m = _MDY_RE.match(text or "")
    if not m:
        return pd.NaT, False
    month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        ts = pd.Timestamp(year=expected_year, month=month, day=day)
    except ValueError:
        return pd.NaT, False
    return ts, (year % 100) != (expected_year % 100)
 
 
# --------------------------------------------------------------------------- #
# 3. Opening / closing dates
# --------------------------------------------------------------------------- #
def clean_opening_closing(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in raw.columns}
    c_season, c_open, c_close = cols["season"], cols["opening date"], cols["closing date"]
    c_days = next((c for k, c in cols.items() if "total days" in k), None)
 
    rows = []
    for _, r in raw.iterrows():
        start = parse_season(r[c_season])
        if start is None:                      # footer / stray row
            continue
        open_txt, open_note = split_note(r[c_open])
        close_txt, close_note = split_note(r[c_close])
        days_txt, _ = split_note(r[c_days]) if c_days else ("", "")
 
        opening, open_fixed = parse_mdy(open_txt, start)
        closing, close_fixed = parse_mdy(close_txt, start + 1)
        if open_fixed or close_fixed:
            log.warning("%d-%02d: page year disagrees with season label (opening=%r, "
                        "closing=%r); using the season label's years",
                        start, (start + 1) % 100, open_txt, close_txt)
        if pd.isna(opening) or pd.isna(closing):
            log.warning("%d-%02d: couldn't parse dates (%r, %r)", start, (start + 1) % 100,
                        open_txt, close_txt)
 
        rows.append({
            "season_start_year": start,
            "opening_date": opening,
            "closing_date": closing,
            "total_days_reported": parse_number(days_txt),
            "date_year_corrected": open_fixed or close_fixed,
            "opening_note": open_note,
            "closing_note": close_note,
        })
 
    df = pd.DataFrame(rows).drop_duplicates("season_start_year", keep="last")
    df = df.sort_values("season_start_year").reset_index(drop=True)
 
    # (closing - opening) in days: same convention as your example table (Alta 163).
    # The page's own "Total Days" column is not always computed the same way, so it is
    # kept separately as total_days_reported.
    df["season_length"] = (df["closing_date"] - df["opening_date"]).dt.days.astype("Int64")
    df["covid_shutdown"] = df["closing_note"].str.contains("covid", case=False)
 
    odd_open = ~df["opening_date"].dt.month.isin([9, 10, 11, 12]) & df["opening_date"].notna()
    odd_close = ~df["closing_date"].dt.month.isin([3, 4, 5, 6]) & df["closing_date"].notna()
    for y in df.loc[odd_open | odd_close, "season_start_year"]:
        log.warning("%d: opening/closing month looks unusual - check the source", y)
 
    df["length_ok"] = df["season_length"].notna() & ~df["covid_shutdown"]
    return df
 
 
# --------------------------------------------------------------------------- #
# 4. Snowfall
# --------------------------------------------------------------------------- #
def detect_repeated_blocks(values, min_run: int = 4, max_lag: int = 20) -> np.ndarray:
    """
    Flag seasons whose value equals the value `lag` seasons earlier for at least
    `min_run` consecutive seasons. Real snowfall never repeats to the half-inch for
    4+ years in a row, so a long exact repeat means a copy/paste artifact.
    The LATER copy is flagged (we can't tell from the numbers which copy is the real one).
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    flagged = np.zeros(n, dtype=bool)
    for lag in range(1, min(max_lag, n - 1) + 1):
        same = np.zeros(n, dtype=bool)
        same[lag:] = v[lag:] == v[:-lag]
        i = 0
        while i < n:
            if same[i]:
                j = i
                while j < n and same[j]:
                    j += 1
                if j - i >= min_run:
                    flagged[i:j] = True
                i = j
            else:
                i += 1
    return flagged
 
 
def clean_snowfall(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in raw.columns}
    c_season = cols["season"]
    c_snow = next(c for k, c in cols.items() if "snowfall" in k)
 
    rows = []
    for _, r in raw.iterrows():
        start = parse_season(r[c_season])
        if start is None:
            continue
        txt, note = split_note(r[c_snow])
        value = parse_number(txt)
        if np.isnan(value):
            log.warning("%d: couldn't parse snowfall %r", start, r[c_snow])
        rows.append({"season_start_year": start, "season_snowfall_in": value,
                     "snowfall_note": note})
 
    df = pd.DataFrame(rows).drop_duplicates("season_start_year", keep="last")
    df = df.sort_values("season_start_year").reset_index(drop=True)
 
    years = df["season_start_year"]
    if (years.diff().dropna() != 1).any():
        log.warning("Snowfall seasons are not consecutive - repeated-block check may miss things")
 
    df["snowfall_repeated_block"] = detect_repeated_blocks(df["season_snowfall_in"])
    if df["snowfall_repeated_block"].any():
        bad = df.loc[df["snowfall_repeated_block"], "season_start_year"]
        log.warning("Snowfall for %d-%d exactly repeats an earlier run of seasons "
                    "(%d values) - likely a copy/paste error in the source; flagged",
                    bad.min(), bad.max() + 1, len(bad))
    df["snowfall_ok"] = df["season_snowfall_in"].notna() & ~df["snowfall_repeated_block"]
    return df
 
 
# --------------------------------------------------------------------------- #
# 5. Merge
# --------------------------------------------------------------------------- #
def build_dataset(opening: pd.DataFrame, snowfall: pd.DataFrame,
                  resort: str = RESORT) -> pd.DataFrame:
    df = opening.merge(snowfall, on="season_start_year", how="outer", validate="one_to_one")
    df = df.sort_values("season_start_year").reset_index(drop=True)
    df["resort"] = resort
    df["season"] = df["season_start_year"].map(lambda y: f"{y}-{y + 1}")
    for flag in ["length_ok", "snowfall_ok"]:           # seasons missing from one page
        df[flag] = df[flag].fillna(False).astype(bool)
    for flag in ["covid_shutdown", "date_year_corrected", "snowfall_repeated_block"]:
        df[flag] = df[flag].fillna(False).astype(bool)
 
    front = ["resort", "season", "season_start_year", "opening_date", "closing_date",
             "season_length", "season_snowfall_in", "length_ok", "snowfall_ok",
             "covid_shutdown", "date_year_corrected", "snowfall_repeated_block",
             "total_days_reported", "opening_note", "closing_note", "snowfall_note"]
    return df[front]
 
 
# --------------------------------------------------------------------------- #
# 6. Wrapper + CLI
# --------------------------------------------------------------------------- #
def load_loveland_history(use_cache: bool = False, opening_html: str | Path | None = None,
                          snowfall_html: str | Path | None = None, save: bool = True):
    """Return (seasons, opening, snowfall) DataFrames."""
    if opening_html:
        o_html = Path(opening_html).read_text(encoding="utf-8")
    else:
        o_html = fetch_html(OPENING_URL, RAW_DIR / "loveland_opening_closing.html", use_cache)
    if snowfall_html:
        s_html = Path(snowfall_html).read_text(encoding="utf-8")
    else:
        s_html = fetch_html(SNOWFALL_URL, RAW_DIR / "loveland_snowfall_history.html", use_cache)
 
    opening = clean_opening_closing(find_table(o_html, "opening date", "closing date"))
    snowfall = clean_snowfall(find_table(s_html, "season", "snowfall"))
    seasons = build_dataset(opening, snowfall)
 
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        seasons.to_csv(PROCESSED_DIR / "loveland_seasons_history.csv", index=False)
        opening.to_csv(PROCESSED_DIR / "loveland_opening_closing.csv", index=False)
        snowfall.to_csv(PROCESSED_DIR / "loveland_snowfall_by_season.csv", index=False)
        log.info("Wrote CSVs to %s", PROCESSED_DIR)
    return seasons, opening, snowfall
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--use-cache", action="store_true", help="reuse data/raw/ copies")
    ap.add_argument("--opening-html", help="parse a saved opening/closing page")
    ap.add_argument("--snowfall-html", help="parse a saved snowfall page")
    args = ap.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    seasons, _, _ = load_loveland_history(args.use_cache, args.opening_html, args.snowfall_html)
 
    both = seasons["length_ok"] & seasons["snowfall_ok"]
    print(f"\n{len(seasons)} seasons total | {seasons['opening_date'].notna().sum()} with dates | "
          f"{seasons['season_snowfall_in'].notna().sum()} with snowfall | "
          f"{both.sum()} with both and no flags")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(seasons.drop(columns=["resort", "season_start_year", "opening_note",
                                    "closing_note", "snowfall_note"]).tail(8))
 
 
if __name__ == "__main__":
    main()
 
