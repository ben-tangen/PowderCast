#!/usr/bin/env python3
"""
build_ski_dataset.py

Combine the Kaggle Epic/Ikon resort CSV with OnTheSnow season snowfall into one
resort-season table:

    resort, season, latitude, longitude, base_elevation_ft, top_elevation_ft,
    elevation_difference_ft, opening_date, closing_date, season_length,
    season_snowfall_in            (+ a few helper columns, listed below)

Sources
-------
  Kaggle CSV    resorts, base/top elevation, opening/closing dates for 2021-22..2023-24
  OnTheSnow     season snowfall (about 13 seasons per resort)
  Nominatim     latitude / longitude (OpenStreetMap geocoder, cached)

Resorts that can't be found on OnTheSnow are left out of the dataset (they are listed,
with the reason, in the match report). Because dates only exist for 3 seasons but
snowfall for ~13, rows outside 2021-22..2023-24 have empty date columns; use
has_dates / has_snowfall, or the *_complete.csv file, to pick what you need.

Usage
-----
    python build_ski_dataset.py --contact-email you@example.com
    python build_ski_dataset.py --limit 5 --no-geocode      # quick smoke test
    python build_ski_dataset.py --include-southern          # see note below

Outputs (data/processed/)
    ski_resort_seasons.csv           every resort-season we have snowfall and/or dates for
    ski_resort_seasons_complete.csv  only rows with dates + snowfall + coordinates
    ski_resort_match_report.csv      one row per CSV resort: matched? why not? coords?

Fixing mismatches (all optional CSVs in data/raw/, re-runs are fast thanks to caching)
    onthesnow_overrides.csv    columns: resort,path   e.g.  Alta Ski Area,utah/alta-ski-area
    coordinate_overrides.csv   columns: resort,latitude,longitude

Southern Hemisphere resorts (Australia, New Zealand, Chile) are excluded by default:
in the CSV their "2021-2022" season holds June-Oct 2022 dates, and I couldn't verify how
OnTheSnow labels those seasons, so a one-year misalignment is possible.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import parse_onthesnow as ots
from get_resort_latitude_longitude import Geocoder

log = logging.getLogger("build")

CSV_NAME = "2024_epic_ikon_resorts_stats.csv"
CSV_CANDIDATES = [Path("data/raw") / CSV_NAME, Path("/data/raw") / CSV_NAME]
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OVERRIDES_PATH = RAW_DIR / "onthesnow_overrides.csv"
COORD_OVERRIDES_PATH = RAW_DIR / "coordinate_overrides.csv"
SOUTHERN_COUNTRIES = {"Australia", "New Zealand", "Chile", "Argentina", "South Africa"}

FINAL_COLUMNS = [
    "resort", "season", "latitude", "longitude", "base_elevation_ft", "top_elevation_ft",
    "elevation_difference_ft", "opening_date", "closing_date", "season_length",
    "season_snowfall_in",
    # helpers
    "season_start_year", "state_or_province", "country", "hemisphere",
    "has_dates", "has_snowfall", "coord_source",
]
_METHOD_RANK = {"override": 0, "exact": 1, "fuzzy": 2, "guess": 3}


# --------------------------------------------------------------------------- #
# 1. Kaggle CSV -> resort table + long dates table
# --------------------------------------------------------------------------- #
def find_csv(explicit: str | None = None) -> Path:
    candidates = [Path(explicit)] if explicit else CSV_CANDIDATES
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Couldn't find the Kaggle CSV. Tried: "
                            + ", ".join(map(str, candidates)) + ". Pass --csv PATH.")


def region_slugs(state: str, country: str) -> list[str]:
    """OnTheSnow region(s) to look in: states / provinces for US & Canada, else the country."""
    if country in ("United States", "Canada"):
        regions = [ots.slugify(s) for s in state.split("/") if s.strip()]
        return regions + (["canada"] if country == "Canada" else [])
    return [ots.slugify(country)]


def load_kaggle(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (resorts, dates_long). `resort` is a unique label (state added to duplicates)."""
    raw = pd.read_csv(path)
    resorts = pd.DataFrame({
        "csv_name": raw["ski_resort_name"].str.strip(),
        "state_or_province": raw["state_or_province"].str.strip(),
        "state_abbr": raw["state_or_province_abbreviation"].str.strip(),
        "country": raw["country"].str.strip(),
        "base_elevation_ft": raw["base_elevation_ft"],
        "top_elevation_ft": raw["top_elevation_ft"],
    })
    resorts["resort"] = resorts["csv_name"]
    dup = resorts["csv_name"].duplicated(keep=False)           # e.g. two 'Hidden Valley's
    resorts.loc[dup, "resort"] = resorts["csv_name"] + " (" + resorts["state_abbr"] + ")"
    resorts["elevation_difference_ft"] = resorts["top_elevation_ft"] - resorts["base_elevation_ft"]
    resorts["hemisphere"] = np.where(resorts["country"].isin(SOUTHERN_COUNTRIES), "S", "N")
    resorts["regions"] = [region_slugs(s, c) for s, c in
                          zip(resorts["state_or_province"], resorts["country"])]
    if "vertical_ft" in raw and (raw["vertical_ft"] != resorts["elevation_difference_ft"]).any():
        log.warning("vertical_ft != top - base for %d resort(s); using top - base",
                    int((raw["vertical_ft"] != resorts["elevation_difference_ft"]).sum()))
    if dup.any():
        log.info("Disambiguated duplicate names: %s", sorted(resorts.loc[dup, "resort"]))

    seasons = sorted({m.group(1) for c in raw.columns
                      if (m := re.match(r"^(\d{4}-\d{4})_opening_day$", c))})
    frames = []
    for season in seasons:
        f = pd.DataFrame({
            "resort": resorts["resort"],
            "season_start_year": int(season[:4]),
            "opening_date": pd.to_datetime(raw[f"{season}_opening_day"], errors="coerce"),
            "closing_date": pd.to_datetime(raw[f"{season}_closing_day"], errors="coerce"),
        })
        # (closing - opening).days: the convention in your target table (Alta: 163).
        f["season_length"] = (f["closing_date"] - f["opening_date"]).dt.days.astype("Int64")
        reported = raw.get(f"{season}_length_days")
        if reported is not None:
            bad = f["season_length"].notna() & (f["season_length"] != reported)
            if bad.any():
                log.warning("%s: CSV length_days disagrees with dates for %s", season,
                            f.loc[bad, "resort"].tolist())
        frames.append(f)
    return resorts, pd.concat(frames, ignore_index=True)


def read_pairs(path: Path, key: str, value_cols: list[str]) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    missing = {key, *value_cols} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if len(value_cols) == 1:
        return dict(zip(df[key], df[value_cols[0]]))
    return {k: tuple(v) for k, v in zip(df[key], df[value_cols].itertuples(index=False))}


# --------------------------------------------------------------------------- #
# 2. Resolve every resort on OnTheSnow and scrape its snowfall
# --------------------------------------------------------------------------- #
def resolve_all(resorts: pd.DataFrame, fetcher: ots.Fetcher, overrides: dict,
                include_southern: bool) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    listing_cache: dict[str, list] = {}
    snow_frames: dict[str, pd.DataFrame] = {}
    rows = []

    for r in resorts.itertuples():
        row = {"resort": r.resort, "state_or_province": r.state_or_province,
               "country": r.country, "hemisphere": r.hemisphere, "status": "",
               "onthesnow_path": "", "match_method": "", "match_score": np.nan,
               "closest_listing": "", "n_snowfall_seasons": 0}
        if r.hemisphere == "S" and not include_southern:
            row["status"] = "excluded_southern_hemisphere"
            rows.append(row)
            continue

        if r.resort in overrides:
            match = ots.Match(str(overrides[r.resort]).strip("/"), "override", 1.0)
        else:
            match = ots.resolve_resort(r.csv_name, r.regions, fetcher, listing_cache)
        row.update(onthesnow_path=match.path or "", match_method=match.method,
                   match_score=match.score, closest_listing=match.best_guess)

        if not match.path:
            row["status"] = "no_match"
        else:
            snow = ots.get_snowfall(fetcher, match.path)
            if snow is None or snow.empty:
                row["status"] = "no_snowfall_table"
            else:
                row["status"] = "matched"
                row["n_snowfall_seasons"] = len(snow)
                snow_frames[r.resort] = snow
        rows.append(row)

    report = pd.DataFrame(rows)

    # Two CSV resorts resolving to the same OnTheSnow page -> keep only the best match.
    ok = report[report["status"] == "matched"].copy()
    ok["rank"] = ok["match_method"].map(_METHOD_RANK)
    for path, grp in ok.groupby("onthesnow_path"):
        if len(grp) > 1:
            grp = grp.sort_values(["rank", "match_score"], ascending=[True, False])
            for loser in grp.index[1:]:
                report.loc[loser, "status"] = "duplicate_onthesnow_page"
                snow_frames.pop(report.loc[loser, "resort"], None)
                log.warning("%s and %s both map to %s; keeping %s only",
                            grp.iloc[0]["resort"], report.loc[loser, "resort"], path,
                            grp.iloc[0]["resort"])
    return report, snow_frames


# --------------------------------------------------------------------------- #
# 3. Coordinates
# --------------------------------------------------------------------------- #
def add_coordinates(report: pd.DataFrame, resorts: pd.DataFrame, geocoder: Geocoder | None,
                    coord_overrides: dict) -> pd.DataFrame:
    report = report.copy()
    for col, default in [("latitude", np.nan), ("longitude", np.nan), ("coord_source", ""),
                         ("coord_state_match", None), ("coord_display_name", "")]:
        report[col] = default
    info = resorts.set_index("resort")

    for i, row in report[report["status"] == "matched"].iterrows():
        if row["resort"] in coord_overrides:
            lat, lon = coord_overrides[row["resort"]]
            report.loc[i, ["latitude", "longitude", "coord_source"]] = [lat, lon, "override"]
        elif geocoder is None:
            report.loc[i, "coord_source"] = "skipped"
        else:
            r = info.loc[row["resort"]]
            hit = geocoder.geocode(r["csv_name"], r["state_or_province"], r["country"])
            if hit:
                report.loc[i, ["latitude", "longitude", "coord_source",
                               "coord_display_name"]] = [hit["latitude"], hit["longitude"],
                                                         "nominatim", hit["coord_display_name"]]
                report.at[i, "coord_state_match"] = hit["coord_state_match"]
            else:
                report.loc[i, "coord_source"] = "not_found"
    for col in ("latitude", "longitude"):
        report[col] = pd.to_numeric(report[col])
    return report


# --------------------------------------------------------------------------- #
# 4. Assemble the final table
# --------------------------------------------------------------------------- #
def assemble(resorts: pd.DataFrame, dates_long: pd.DataFrame,
             snow_frames: dict[str, pd.DataFrame], report: pd.DataFrame) -> pd.DataFrame:
    matched = report.loc[report["status"] == "matched", "resort"]
    if matched.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    snow_long = pd.concat([f.assign(resort=name) for name, f in snow_frames.items()
                           if name in set(matched)], ignore_index=True)
    dates = dates_long[dates_long["resort"].isin(matched)]
    df = dates.merge(snow_long, on=["resort", "season_start_year"],
                     how="outer", validate="one_to_one")

    attrs = (resorts.set_index("resort")[["base_elevation_ft", "top_elevation_ft",
                                          "elevation_difference_ft", "state_or_province",
                                          "country", "hemisphere"]]
             .join(report.set_index("resort")[["latitude", "longitude", "coord_source"]]))
    df = df.join(attrs, on="resort")
    df["season"] = df["season_start_year"].map(lambda y: f"{y}-{y + 1}")
    df["has_dates"] = df["opening_date"].notna() & df["closing_date"].notna()
    df["has_snowfall"] = df["season_snowfall_in"].notna()
    df["season_length"] = df["season_length"].astype("Int64")
    return (df[FINAL_COLUMNS].sort_values(["resort", "season_start_year"])
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# 5. Orchestration
# --------------------------------------------------------------------------- #
def build_dataset(csv_path: Path, fetcher: ots.Fetcher, geocoder: Geocoder | None,
                  include_southern: bool = False, limit: int | None = None,
                  out_dir: Path | None = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    resorts, dates_long = load_kaggle(csv_path)
    if limit:
        resorts = resorts.head(limit)
        dates_long = dates_long[dates_long["resort"].isin(resorts["resort"])]
    log.info("%d resorts in the CSV", len(resorts))

    overrides = read_pairs(OVERRIDES_PATH, "resort", ["path"])
    coord_overrides = read_pairs(COORD_OVERRIDES_PATH, "resort", ["latitude", "longitude"])

    report, snow_frames = resolve_all(resorts, fetcher, overrides, include_southern)
    report = add_coordinates(report, resorts, geocoder, coord_overrides)
    final = assemble(resorts, dates_long, snow_frames, report)

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        complete = final[final["has_dates"] & final["has_snowfall"]
                         & final["latitude"].notna() & final["longitude"].notna()]
        final.to_csv(out_dir / "ski_resort_seasons.csv", index=False)
        complete.to_csv(out_dir / "ski_resort_seasons_complete.csv", index=False)
        report.to_csv(out_dir / "ski_resort_match_report.csv", index=False)
        log.info("Wrote %s", out_dir)
    return final, report


def print_summary(final: pd.DataFrame, report: pd.DataFrame) -> None:
    print("\nResort status:")
    print(report["status"].value_counts().to_string())
    left_out = report[~report["status"].isin(["matched"])]
    if len(left_out):
        shown = 40
        print("\nLeft out of the dataset (full list + closest listings: ski_resort_match_report.csv):")
        for r in left_out.head(shown).itertuples():
            hint = f"  closest: {r.closest_listing}" if r.closest_listing else ""
            print(f"  - {r.resort} [{r.status}]{hint}")
        if len(left_out) > shown:
            print(f"  ... and {len(left_out) - shown} more")
    if final.empty:
        print("\nNo matched resorts.")
        return
    comp = final["has_dates"] & final["has_snowfall"] & final["latitude"].notna()
    print(f"\nFinal table: {len(final)} resort-seasons from {final['resort'].nunique()} resorts "
          f"| with snowfall: {int(final['has_snowfall'].sum())} "
          f"| with dates: {int(final['has_dates'].sum())} | fully complete: {int(comp.sum())}")
    bad = report[(report["coord_source"] == "nominatim") & (report["coord_state_match"] == False)]  # noqa: E712
    nocoord = report[(report["status"] == "matched") & report["latitude"].isna()]
    if len(bad):
        print(f"Coordinates whose state didn't match the CSV (check these): {bad['resort'].tolist()}")
    if len(nocoord):
        print(f"Matched resorts with NO coordinates: {nocoord['resort'].tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[1].split()))
    ap.add_argument("--csv", help=f"path to {CSV_NAME} (default: data/raw/, then /data/raw/)")
    ap.add_argument("--contact-email", default="", help="sent in the User-Agent to Nominatim")
    ap.add_argument("--refresh", action="store_true", help="ignore cached pages / geocodes")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between OnTheSnow requests")
    ap.add_argument("--include-southern", action="store_true",
                    help="include Australia / NZ / Chile (season labels may be off by a year)")
    ap.add_argument("--no-geocode", action="store_true", help="skip latitude/longitude lookup")
    ap.add_argument("--limit", type=int, help="only process the first N resorts (testing)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fetcher = ots.Fetcher(delay=args.delay, use_cache=not args.refresh)
    geocoder = None if args.no_geocode else Geocoder(contact=args.contact_email,
                                                     use_cache=not args.refresh)
    try:
        final, report = build_dataset(find_csv(args.csv), fetcher, geocoder,
                                      args.include_southern, args.limit)
    except ots.FetchBlocked as exc:
        raise SystemExit(f"ERROR: {exc}")
    print_summary(final, report)


if __name__ == "__main__":
    main()
