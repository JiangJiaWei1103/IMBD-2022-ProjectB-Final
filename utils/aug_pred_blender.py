"""Augmented prediction blender."""
import numpy as np
import pandas as pd


class AugPredBlender(object):
    """Blender for blending predicting results obtained from models
    trained on different window-chunks of data in the same layer.

    With data augmentation, one layer can have more than one prediction.

    Parameters:
        strategy: blending strategy, the choices are as follows:
            {"mean", "wt"} -> wt strategy
    """

    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy

    def blend(self, layer_ids: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        y_pred_da = pd.DataFrame(y_pred, columns=["y_pred"])
        y_pred_da["layer"] = layer_ids.values

        if self.strategy == "mean":
            y_pred_blended = y_pred_da.groupby("layer")["y_pred"].mean().values

        return y_pred_blended
