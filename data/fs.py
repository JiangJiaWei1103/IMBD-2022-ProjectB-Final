"""Feature selector."""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression, mutual_info_regression
from sklearn.preprocessing import QuantileTransformer

SPECIAL_ENTRIES = [np.inf, -np.inf, np.nan]


class FeatureSelector(object):
    """Simple feature selector to reduce feature dim.


    Attributes:
        feats_orig_: original features before selection
        slc_mask_: adjustable selection mask
    """

    feats_orig_: pd.Index
    slc_mask_: np.ndarray
    feats_slc_: List[str]

    def __init__(
        self,
        X_shape: Tuple[int, int],
        n_quantiles: Optional[int] = None,
        var_thres: Optional[float] = None,
        kbest_score_fn: Optional[str] = None,
        kbest_k: Optional[int] = None,
    ):
        self.X_shape = X_shape
        self.slc_mask_ = np.ones(X_shape[1])

        self.n_quantiles = n_quantiles
        self.var_thres = var_thres
        self.kbest_score_fn = kbest_score_fn
        self.kbest_k = kbest_k

    def run(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """Run feature selection pipeline."""
        self.feats_orig_ = X.columns

        self._drop_zero_cols(X)
        # ===
        X = X.replace(SPECIAL_ENTRIES, 0)

        scl = QuantileTransformer(n_quantiles=self.n_quantiles)
        X = pd.DataFrame(scl.fit_transform(X), columns=X.columns)
        # ===
        if self.n_quantiles is not None and self.var_thres is not None:
            self._var_thres(X)
        if self.kbest_score_fn is not None and self.kbest_k is not None:
            self._kbest(X, y)

        logging.info("-" * 50)
        logging.info(f"{np.sum(self.slc_mask_)} features are selected.")
        X_slc = X[self.feats_orig_[self.slc_mask_]]
        self.feats_slc_ = X_slc.columns.to_list()

        return X_slc

    def _drop_zero_cols(self, X: pd.DataFrame) -> None:
        """Drop columns with all entries to be zeros."""
        logging.info("Start FS via dropping zero columns...")
        non_zeros_mask = (((X == 0).sum() / len(X)) != 1).values
        self._update_slc_mask(non_zeros_mask)

    def _var_thres(self, X: pd.DataFrame) -> None:
        """Select features via variance threhold."""
        logging.info(f"Start FS via VarianceThreshold({self.var_thres})...")
        X_orig_cols = X.columns[self.slc_mask_]
        X_orig = X[X_orig_cols]

        #         scl = QuantileTransformer(n_quantiles=self.n_quantiles)
        #         X_scl = pd.DataFrame(scl.fit_transform(X_orig), columns=X_orig.columns)
        X_scl = X_orig

        fs = VarianceThreshold(threshold=self.var_thres)
        fs.fit(X_scl)
        feats_slc = fs.get_feature_names_out(X_orig_cols)
        self._update_slc_mask(self.feats_orig_.isin(feats_slc))

    def _kbest(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        """Select features via scoring functions."""
        logging.info(f"Start FS via SelectKBest({self.kbest_score_fn}, k={self.kbest_k})...")
        X_orig_cols = X.columns[self.slc_mask_]
        X_orig = X[X_orig_cols]

        #         scl = StandardScaler()
        #         X_scl = pd.DataFrame(scl.fit_transform(X_orig), columns=X_orig.columns)
        X_scl = X_orig

        if self.kbest_score_fn == "f":
            score_fn = f_regression
        elif self.kbest_score_fn == "m":
            score_fn = mutual_info_regression
        fs = SelectKBest(score_fn, k=self.kbest_k)
        fs.fit(X_scl, y)
        feats_slc = fs.get_feature_names_out(X_orig_cols)
        self._update_slc_mask(self.feats_orig_.isin(feats_slc))

    def _update_slc_mask(self, slc_mask_new: np.ndarray) -> None:
        """Update feature selection mask."""
        n_feats_orig = np.sum(self.slc_mask_)
        self.slc_mask_ = np.logical_and(self.slc_mask_, slc_mask_new.astype(np.int32))
        n_feats_slc = np.sum(self.slc_mask_)
        logging.info(f"-> {int(n_feats_orig - n_feats_slc)} features are dropped.")
