"""Utilities for evaluation."""
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from metadata import GP1_LEN, GP2_LEN, N_PROC_LAYERS, TARGET
from paths import RAW_DATA_PATH


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Derive and return RMSE."""
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


class Evaluator(object):
    """Evaluator deriving, summarizing and logging evaluation results.

    Parameters:
        dataset: name of the dataset, the choices are as folows:
            {"train1", "train2", "mix"}
        eval_range: evaluation range
            *Note: If chunk-aware modeling is used, then `eval_range`
                should be limited
    """

    def __init__(self, dataset: str = "train1", eval_range: List[str] = ["all", "gp1", "gp2"]):
        self.dataset = dataset
        self.gp1_len = GP1_LEN[dataset]
        self.gp2_len = GP2_LEN[dataset]
        self.y = pd.read_csv(os.path.join(RAW_DATA_PATH, dataset, "wear.csv"))[TARGET]
        self.eval_range = eval_range

        assert (
            N_PROC_LAYERS[dataset] == self.gp1_len + self.gp2_len
        ), "#Processing layers isn't equal to len(gp1) + len(gp2)."

    def evaluate(self, preds: List[np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """Run evaluation.

        Note we can add more eval ranges to do more fine-grained
        evaluation (e.g., fixing layer k).
        """
        final_scores = defaultdict(list)
        for pred in preds:
            if "all" in self.eval_range:
                rmse_all = rmse(self.y, pred)
                final_scores["all"].append(rmse_all)
            if "gp1" in self.eval_range:
                rmse_gp1 = rmse(self.y[: self.gp1_len], pred[: self.gp1_len])
                final_scores["gp1"].append(rmse_gp1)
            if "gp2" in self.eval_range:
                rmse_gp2 = rmse(self.y[-self.gp2_len :], pred[-self.gp2_len :])
                final_scores["gp2"].append(rmse_gp2)

        final_scores_agg = {}
        for eval_range, rmses in final_scores.items():
            final_scores_agg[eval_range] = (np.mean(rmses), np.std(rmses))

        return final_scores_agg  # type: ignore
