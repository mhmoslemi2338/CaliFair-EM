# Official implementation of GUIDE-AI @ SIGMOD paper "Threshold-Independent Fair Matching through Score Calibration"

## Abstract
Entity Matching (EM) is a critical task in numerous fields, such as healthcare, finance, and public administration, as it identifies records that refer to the same entity within or across different databases. EM faces considerable challenges, particularly with false positives and negatives. These are typically addressed by generating matching scores and apply thresholds to balance false positives and negatives in various contexts. However, adjusting these thresholds can affect the fairness of the outcomes, a critical factor that remains largely overlooked in current fair EM research. The existing body of research on fair EM tends to concentrate on static thresholds, neglecting their critical impact on fairness. To address this, we introduce a new approach in EM using recent metrics for evaluating biases in score based binary classification, particularly through the lens of distributional parity. This approach enables the application of various bias metrics like equalized odds, equal opportunity, and demographic parity without depending on threshold settings. Our experiments with leading matching methods reveal potential biases, and by applying a calibration technique for EM scores using Wasserstein barycenters, we not only mitigate these biases but also preserve accuracy across real world datasets. This paper contributes to the field of fairness in data cleaning, especially within EM, which is a central task in data cleaning, by promoting a method for generating matching scores that reduce biases across different thresholds.

## Data Directory

You can find all the data we used in the `DATA` directory.

The dataset are from the paper Deep Learning for Entity Matching: A Design Space Exploration, SIGMOD 2018 at [https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)

## Implementation Details

In each of the directories for `DITTO`, `DeepMatcher`, `EMTransformer`, `HierGAT`, `HierMatcher`, and `Magellan`, you will find the implementation for each method and instructions for obtaining the results. Each directory contains a Python script named starting with `Train_`. You can use this script to retrain the network. After training, the score for the test data will be automatically saved in the `SCORES` directory.

## Regenerating Experiments

You can regenerate the experiments from the `experiments.ipynb` file, which utilizes the scores in the `SCORES` directory. This notebook also saves some variables in `.pkl` format and saves the final results and measurements in `.csv` format, as well as figures in `.pdf` format, in the `FIGURES` directory.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{moslemi2024threshold,
  title={Threshold-Independent Fair Matching through Score Calibration},
  author={Moslemi, Mohammad Hossein and Milani, Mostafa},
  booktitle={Proceedings of the Conference on Governance, Understanding and Integration of Data for Effective and Responsible AI},
  pages={40--44},
  year={2024}
}
