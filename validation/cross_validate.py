"""
Cross validation core logic.

This file contains the core logic of running cross validation.
"""
import logging
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._split import _BaseKFold

from modeling.build import build_models
from utils.aug_pred_blender import AugPredBlender
from utils.evaluating import Evaluator, rmse
from utils.shuffle_gpkf import ShuffleGroupKFold
from utils.traits import is_gbdt_instance  # type: ignore


class MultiSeedCVWrapper(object):
    """Wrapper for running CV using multiple random seeds.

    The wrapper also supports passing multiple data splitting iterables
    to facilitate parallel modeling comparisons.

    Parameters:
        n_seeds: number of random seeds used to split the dataset
        n_folds: number of folds for a single CV run
        kfs: pre-defined CV strategies
        verbose: whether to log message along CV process
        mix_aug: whether to train model with aug data all at once
            *Note: Models are trained on data chunks mixed together
    """

    def __init__(
        self,
        n_seeds: int = 20,
        n_folds: int = 10,
        kfs: Optional[List[BaseCrossValidator]] = None,
        verbose: bool = True,
        mix_aug: bool = False,
    ):
        self.n_seeds = n_seeds
        self.n_folds = n_folds
        self.kfs = kfs
        self.mix_aug = mix_aug
        self.verbose = verbose

        if n_folds != -1:
            self.seeds = MultiSeedCVWrapper._gen_seeds(n_seeds)
            if mix_aug:
                logging.info(f"ShuffleGroupKFolds with {n_seeds} random seeds are used...\n")
                base_kf = ShuffleGroupKFold
            else:
                logging.info(f"Naive KFolds with {n_seeds} random seeds are used...\n")
                base_kf = KFold

            self.kfs = [base_kf(n_folds, shuffle=True, random_state=seed) for seed in self.seeds]
        else:
            self.kfs = [HardKFold()]

    @staticmethod
    def _gen_seeds(n_seeds: int) -> List[int]:
        """Generate n random seeds."""
        return random.sample(range(2**32 - 1), n_seeds)

    def run_cv(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        model_name: str = "rf",
        model_params: Dict[str, Any] = {},
        fit_params: Dict[str, Any] = {},
        evaluator: Evaluator = Evaluator("train1"),
        group: Optional[pd.Series] = None,
    ) -> Tuple[List[np.ndarray], List[List[float]], List[List[float]], List[List[BaseEstimator]], pd.DataFrame]:
        """Run cross-validation using multiple data splitting schemes."""
        oof_preds = []
        tr_rmses = []
        val_rmses = []
        models = []
        feat_imps = pd.DataFrame()

        for i, kf in enumerate(self.kfs):
            if self.verbose:
                logging.info(f"Start CV Round {i}...")

            models_seed = build_models(model_name, model_params, kf.get_n_splits())
            (oof_pred_seed, tr_rmses_seed, val_rmses_seed, feat_imps_seed) = self._run_cv_single_seed(
                X, y, kf, models_seed, fit_params, evaluator, group
            )

            tr_rmses.append(tr_rmses_seed)
            val_rmses.append(val_rmses_seed)
            oof_preds.append(oof_pred_seed)
            models.append(models_seed)
            if model_name == "lgbm":
                feat_imps = pd.concat([feat_imps, feat_imps_seed])

        self._log_prf_summary(tr_rmses, val_rmses, evaluator.evaluate(oof_preds))

        if model_name == "lgbm":
            feat_imps.reset_index(inplace=True)  # Reset `feature` column
        else:
            assert len(feat_imps) == 0, "Feature importance isn't supported."

        return oof_preds, tr_rmses, val_rmses, models, feat_imps

    def _run_cv_single_seed(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        kf: BaseCrossValidator,
        models: List[BaseEstimator],
        fit_params: Dict[str, Any] = {},
        evaluator: Evaluator = Evaluator("train1"),
        group: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, List[float], List[float], pd.DataFrame]:
        """Run cross-validation using a single data splitting scheme."""
        oof_pred = np.zeros(len(y))
        tr_rmses, val_rmses = [], []
        feat_imps = pd.DataFrame()

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X, groups=group)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # Fit the model
            models[fold].fit(X_tr, y_tr, **fit_params)

            # Perform OOF prediction
            oof_pred[val_idx] = models[fold].predict(X_val)

            # Derive RMSE on wear difference
            tr_rmses.append(rmse(y_tr, models[fold].predict(X_tr)))
            val_rmses.append(rmse(y_val, oof_pred[val_idx]))

            if is_gbdt_instance(models[fold], "lgbm"):
                feat_imps_fold = self._get_feat_imp(models[fold], X.columns)
                feat_imps = pd.concat([feat_imps, feat_imps_fold], ignore_index=True)

        if group is not None:
            oof_pred = AugPredBlender(strategy="mean").blend(group, oof_pred)

        # Report peformance summary
        if self.verbose:
            final_score = evaluator.evaluate([oof_pred])  # Pass single pred in list
            logging.info("=====Performance Summary of Single Seed=====")
            logging.info(f"Train RMSE: {np.mean(tr_rmses):.5f} +- {np.std(tr_rmses):.5f}")
            logging.info(f"Val RMSE: {np.mean(val_rmses):.5f} +- {np.std(val_rmses):.5f}")
            logging.info(f"Final score: {final_score}\n")

        if len(feat_imps) > 0:
            # Take average over folds
            feat_imps = feat_imps.groupby("feature").mean()

        return oof_pred, tr_rmses, val_rmses, feat_imps

    def _get_feat_imp(
        self,
        model: BaseEstimator,
        feat_names: List[str],
    ) -> pd.DataFrame:
        """Generate and return feature importance DataFrame.

        Parameters:
            model: well-trained estimator
            feat_names: list of feature names

        Return:
            feat_imp: feature importance
        """
        feat_imp = pd.DataFrame(feat_names, columns=["feature"])
        feat_imp["gain"] = model.booster_.feature_importance("gain")
        feat_imp["split"] = model.booster_.feature_importance("split")

        return feat_imp

    def _log_prf_summary(
        self,
        tr_rmses: List[List[float]],
        val_rmses: List[List[float]],
        final_score: Dict[str, Tuple[float, float]],
    ) -> None:
        """Log performance summary for multiple CV rounds."""
        tr_rmses_avg = np.array(tr_rmses).mean(axis=1).mean()
        tr_rmses_std = np.array(tr_rmses).mean(axis=1).std()
        val_rmses_avg = np.array(val_rmses).mean(axis=1).mean()
        val_rmses_std = np.array(val_rmses).mean(axis=1).std()

        logging.info("≡≡≡≡≡≡≡ Overall Performance Summary ≡≡≡≡≡≡≡")
        logging.info(f"Train RMSE: {tr_rmses_avg:.5f} ± {tr_rmses_std:.5f}")
        logging.info(f"Val RMSE: {val_rmses_avg:.5f} ± {val_rmses_std:.5f}")
        logging.info("==> Final score <==")
        for eval_range, prf in final_score.items():
            logging.info(f"{eval_range.upper()}: {prf[0]:.5f} ± {prf[1]:.5f}")
        logging.info("-" * 50)


class HardKFold(_BaseKFold):
    """Manual data splitting scheme with higher difficulty."""

    def __init__(self) -> None:
        super().__init__(n_splits=10, shuffle=False, random_state=None)

    def split(
        self, X: pd.DataFrame, groups: Optional[pd.Series] = None, y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        MANUAL_SPLIT = [
            [32, 33, 34, 35, 36],
            [31, 30, 29, 28, 37],
            [24, 25, 26, 27, 38],
            [23, 22, 21, 20, 39],
            [16, 17, 18, 19, 40],
            [15, 14, 13, 12, 41],
            [9, 10, 11, 42],
            [6, 7, 8, 43],
            [3, 4, 5, 44],
            [0, 1, 2, 45],
        ]

        for i, val_idx in enumerate(MANUAL_SPLIT):
            tr_idx = set(range(46)).difference(set(val_idx))
            tr_idx = np.array(list(tr_idx))

            yield tr_idx, np.array(val_idx)
