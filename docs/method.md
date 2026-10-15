# Method

This document contains the proposed method for gathering data, predicting attributes, and more.

## Gathering Data

See the [datasets document](datasets.md) for more information on where we're sourcing our data from.
When putting together data for a model to train/test on, each season of a ski resort should be structured like so:

(DATA MODEL IS SUBJECT TO CHANGE)

| Field  | Example value  |
|---|---|
| Resort  | Alta  |
| Season  | 2024-2025  |
| **PRE-SEASON DATA**  |   |
| Pre-season ONI/PDO/PNA/AO/NAO data from May-November | TBD  |
| **SEASON DATA**| |
| Opening Date  | November 5th, 2024  |
| Closing Date  | April 17th, 2025  |
| Season Length  | 163  |
| Season Snowfall  | 222 Inches  |


More pre-season data may be added, and some season data may be removed from this.
But essentially, for each season of a resort, a row structured like the above data should be created.
This data can be used to train the models.
Every model's goal will be to take the Pre-Season data and use it to predict the season data (specifically the snowfall and season length).

## What we're predicting

Our goal is to train models to use weather data leading up to a ski season (so weather from March - October) to predict the following two aspects of a ski resort's season:

- Total snowfall the resort receives
- Season length
  - This is the total number of days that the resort is open for the season

We can output predictions for this data with multiple methods.
Firstly, we could predict both the snowfall and season length individually.
This is easier, but will require predicting multiple parameters
Secondly, we could combine total snowfall and season length into one metric, a *Season Quality* score. This would be more difficult, as it would require finding a good way to balance the two pieces of data into one score, but it would make the training process simpler.

## Experiments

TBD.
Theoretically we want to experiment with predicting season length/snowfall with a bunch of different ML/NN methods.
I think it would be best to use Linear Regression as a baseline, and then experiment with using different parameters, weights, and throwing neural networks into the mix.