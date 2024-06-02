import pandas as pd
import numpy as np
from typing import Literal

import sys
sys.path.append('..')
from data_processing.data_import import Elmy_import
from training.helper_func import get_scaler, threshold_percentile

def ensemble_model(filenames, weights=None, filepath=lambda x: f"../data/submission/{x}.csv", newnames=None):
    """
    Only processes submissions containing 1 and -1
    """
    if weights is None:
        weights = np.ones(len(filenames))
    assert np.min(weights) >= 0, "Weights must be nonnegative"
    assert len(filenames) == len(weights) > 2, "weights and filenames must have the same length. There must be at least twwo paths"
    if newnames is not None:
        assert len(filenames) == len(newnames), "newnames and filenames must have the same length"
    votes = pd.concat([Elmy_import(filepath(filename), with_date=False) for filename in filenames], axis=1)
    votes.columns = filenames if newnames is None else newnames
    print(f"Proportion of class 1:\n", (votes==1).mean())
    print(f"Correlation:\n", votes.corr())
    assert votes.abs().min().min() == votes.abs().max().max() == 1, "The submissions must contain only 1 and -1"
    weighted_votes = votes.mul(weights, axis=1)
    ensemble_submission = pd.DataFrame(np.sign(weighted_votes.sum(axis=1)), columns=['spot_id_delta'])
    if newnames is not None:
        submissions = {name: load_submission(filenames[i]) for i, name in enumerate(newnames)}
        for name, submission in submissions.items():
            print(f"Ensemble prediction overlaps by {100*np.mean(submission.values==ensemble_submission.values):.1f}% with {name}")
    return weighted_votes, ensemble_submission

def load_submission(filename,  filepath=lambda x: f"../data/submission/{x}.csv"):
    submission = Elmy_import(filepath(filename), with_date=False)
    submission = pd.DataFrame(submission.values, columns=['spot_id_delta'], index=submission.index)
    return submission

def ensemble_model_2(filenames, normalization: Literal['StandardScaler', 'RobustScaler', 'MinMaxScaler', 'l1'], correction_percentile=52.0, weights=None, filepath=lambda x: f"../data/submission/{x}.csv", newnames=None):
    """
    Processes "true" delta predictions, i.e. real values, not only -1 and 1
    """
    if weights is None:
        weights = np.ones(len(filenames))
    assert np.min(weights) >= 0, "Weights must be nonnegative"
    assert len(filenames) == len(weights) > 2, "weights and filenames must have the same length. There must be at least twwo paths"
    if newnames is not None:
        assert len(filenames) == len(newnames), "newnames and filenames must have the same length"
    votes = pd.concat([Elmy_import(filepath(filename), with_date=False) for filename in filenames], axis=1)
    index = votes.index
    columns = filenames if newnames is None else newnames
    scaler = get_scaler(normalization)
    votes = pd.DataFrame(scaler.fit_transform(votes), index=index, columns=columns)
    print(f"Correlation:\n", votes.corr())
    weighted_votes = votes.mul(weights, axis=1)
    true_ensemble_submission = weighted_votes.sum(axis=1)
    ensemble_submission = pd.DataFrame(threshold_percentile(true_ensemble_submission, correction_percentile), columns=['spot_id_delta'], index=index)
    if newnames is not None:
        submissions = {name: load_submission(filenames[i]) for i, name in enumerate(newnames)}
        for name, submission in submissions.items():
            print(f"Ensemble prediction overlaps by {100*np.mean(submission.values==ensemble_submission.values):.1f}% with {name}")
    return weighted_votes, ensemble_submission   
