# Elmy

This repository addresses the [electricity price prediction data challenge](https://challengedata.ens.fr/participants/challenges/140/) proposed by Elmy in 2024.

## Import the Data

To import the data, run the following command:

```bash
cd data_processing
python data_import.py --interpolate-psp
```

## Train Models

Before training a model, you may need to customize the features to include in the model. You can specify the features using the `--keep "feature1" "feature2"` flag or by modifying the `args2.py` file.

To train a model, run the following command:

```bash
cd training
python train2.py --model CNN --lr 0.002 --batch-size 5 --epochs 100 --visualize-features --num-workers 10
```

You can either train a LSTM, a CNN or a SEAC, which is a variant of a CNN relying on a dynamic feature selection method called [Squeeze-and-Excitation](https://arxiv.org/pdf/1709.01507.pdf).

## Remarks and Further Improvements

1. For this classification problem, the Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) can be used to evaluate the performance of different models. We did not use this metrics to evaluate our models.
2. A model based on attention such as a transformer could outperform our models since we are allowed to use all the data available at day $t$ to predict the delta at any hours of day $t$ (data for day $t$ is already known at day $t-1$ since it is only composed of forecasts).
