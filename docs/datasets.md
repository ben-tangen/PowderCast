# Datasets

## Weather Datasets

- [ONI (El Niño/La Niña)](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt): Ocean temperature anomaly in the central equatorial Pacific; changes over months and is the main preseason predictor.
- [PDO (Pacific Decadal Oscillation)](https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat): Long-lived North Pacific ocean temperature pattern lasting years to decades; strengthens or weakens El Niño's effects.
- [PNA (Pacific–North American pattern)](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii): Jet stream pattern over the Pacific and North America; negative favors storms in the Intermountain West, positive favors dry ridging.
- [AO (Arctic Oscillation)](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii): Strength of the winds holding cold air over the Arctic; negative lets cold air spill south.
- [NAO (North Atlantic Oscillation)](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii): Pressure seesaw between Iceland and the Azores; mainly affects the eastern US and Europe, so it serves as a control for Utah.

## Ski Resort Datasets

- [Loveland Ski Area snowfall archive](https://skiloveland.com/snowfall-history/): Contains snowfall history and season-length history for Loveland Ski Area dating back to 1979. Great dataset that should definitely be used, 
- [Epic and Ikon Ski Resorts Dataset](https://www.kaggle.com/datasets/rummagelabs/ikon-and-epic-resorts-season-statistics/data): Resort metadata and historical opening/closing dates. Dates back to 2021-2022 season.
- [OnTheSnow Snow Reports](https://www.onthesnow.com/utah/skireport): Contains [historical snowfalls](https://www.onthesnow.com/utah/beaver-mountain/historical-snowfall) for many resorts. Good source for total snowfall and biggest snowfall.
- [NRCS Utah Snow Survey/SNOTEL](https://www.nrcs.usda.gov/utah/snow-survey): Historic SWE, snow depth, precipitation, and temperature at mountain stations.

## Summary / Opinions

With our current selection of weather datasets, using datasets close to the ski resorts we're testing would be ideal. 
We want to use the data from those datasets (which should show weather leading up to the ski season) to predict the upcoming snowfall/season length of a resort.
We will most likely find more weather datasets to use as well.

As for ski resorts, I think we should definitely experiment on the Loveland dataset, as it contains everything we would need and has tons of historical data.
 Additionally, testing using a combination of the Epic/Ikon dataset (for open/close dates) and OnTheSnow (for snowfall) would be great for testing a broad amount of resorts.