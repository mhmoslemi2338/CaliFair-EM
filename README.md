# official code for the paper: Threshold-Independent Fair Matching through Score Calibration
https://arxiv.org/abs/2405.20051


## Data Directory

You can find all the data we used in the `DATA` directory.

## Implementation Details

In each of the directories for `DITTO`, `DeepMatcher`, `EMTransformer`, `HierGAT`, `HierMatcher`, and `Magellan`, you will find the implementation for each method and instructions for obtaining the results. Each directory contains a Python script named starting with `Train_`. You can use this script to retrain the network. After training, the score for the test data will be automatically saved in the `SCORES` directory.

## Regenerating Experiments

You can regenerate the experiments from the `experiments.ipynb` file, which utilizes the scores in the `SCORES` directory. This notebook also saves some variables in `.pkl` format and saves the final results and measurements in `.csv` format, as well as figures in `.pdf` format, in the `FIGURES` directory.


