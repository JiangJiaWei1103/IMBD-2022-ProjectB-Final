"""Data processor."""
import logging
import os
import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from metadata import GP1_LEN, PK, PK_AUG, TARGET
from paths import DUMP_PATH, PROC_DATA_PATH, RAW_DATA_PATH

from .fs import FeatureSelector

SPECIAL_ENTRIES = [np.inf, -np.inf, np.nan]


class DataProcessor:
    """Data processor processing raw data and providing access to the
    processed data ready to be fed into models.

    Parameters:
        dataset: name of the dataset, the choices are as follows:
            {"train1", "train2", "test"}
        data_type: type of the data, the choices are as follows:
            {"normal", "aug"}
        mix_aug: if True, models are trained on data chunks mixed
            together
        dp_cfg: hyperparameters of data processor
    """

    _df: pd.DataFrame
    _X: pd.DataFrame
    _y_base: Optional[np.ndarray]
    _y: Optional[np.ndarray]

    def __init__(self, dataset: str, data_type: str, mix_aug: bool, infer: bool, **dp_cfg: Any):
        self.dataset = dataset
        self.data_type = data_type
        self.mix_aug = mix_aug
        self.infer = infer

        # Load X base DataFrame
        data_type = "aug_" if data_type == "aug" else ""
        data_path = os.path.join(PROC_DATA_PATH, f"X_{data_type}{dataset}.csv")
        self._df = pd.read_csv(data_path)

        # Prepare y data
        self._y_base = self._prepare_y()

        # Setup data processing pipeline
        self._dp_cfg = dp_cfg
        self._setup()

    def _prepare_y(self) -> Optional[np.ndarray]:
        """Prepare y data."""
        if self.dataset != "test":
            y = pd.read_csv(os.path.join(RAW_DATA_PATH, f"{self.dataset}/wear.csv"))
            if self.mix_aug:
                y = self._df["layer"].map(y.set_index("Index")[TARGET]).values
            else:
                y = y[TARGET].values

            return y
        else:
            return None

    def _setup(self) -> None:
        """Retrieve all parameters used in data processing pipeline and
        setup feature engineer.
        """
        # Before data splitting
        self.imp_spec_entries = self._dp_cfg["imp_spec_entries"]
        self.fs_cfg = self._dp_cfg["fs"]

        # After data splitting
        self.scale_cfg = self._dp_cfg["scale"]

    def run_before_cv(
        self,
        feats_to_use: Optional[List[str]] = None,
        trafo: Optional[Any] = None,
        gp_id: Optional[int] = None,
        chunk_id: Optional[int] = None,
    ) -> Optional[Any]:
        """Clean and process data before cross validation process.

        Parameters:
            feats_to_use: pre-selected features
                *Note: It's used when `self.infer` is True
            trafo: transfomer
                *Note: It's used when `self.infer` is True
            gp_id: group identifier, either 1 or 2
                *Note: It's used only in chunk-aware modeling
            chunk_id: chunk (slice) identifier
                *Note: It's used only in chunk-aware modeling

        Return:
            trafo: QuantileXFormer
        """
        logging.info("Run data cleaning and processing before data splitting...")

        # Handle special entries
        if self.imp_spec_entries:
            self._df.replace(SPECIAL_ENTRIES, 0, inplace=True)

        # Get pre-chunked data for chunk-aware modeling
        if gp_id is not None and chunk_id is not None:
            gp_sep = GP1_LEN[self.dataset]
            if gp_id == 1:
                # Group1 data is retrieved
                gp_mask = self._df["layer"] <= gp_sep
            elif gp_id == 2:
                # Group2 data is retrieved
                gp_mask = self._df["layer"] > gp_sep
            X = self._df[gp_mask & (self._df["slice"] == chunk_id)]
            if self.dataset != "test":
                y = self._y_base[:gp_sep] if gp_id == 1 else self._y_base[gp_sep:]
        else:
            X = self._df
            if self.dataset != "test":
                y = self._y_base

        pk = PK_AUG if self.data_type == "aug" else PK
        X = X.drop(pk, axis=1)
        # Run feature selection
        if not self.infer:
            fs = FeatureSelector(X.shape, **self.fs_cfg)
            self._X, trafo = fs.run(X, y)
            self._feats_slc = fs.feats_slc_
        else:
            # No need to do feature selection
            assert feats_to_use is not None, "You must provide pre-selected features in inference process."
            # ===
            X = X.replace(SPECIAL_ENTRIES, 0)
            logging.info(f"Use {trafo} to transform data...")
            X = pd.DataFrame(trafo.transform(X), columns=X.columns)
            # ===
            self._X = X[feats_to_use]
            trafo = None  # Reset

        if self.dataset != "test":
            self._y = y
            logging.info(f"X shape: {self._X.shape}, y_shape: {self._y.shape}\n")
        else:
            self._y = None
            logging.info(f"X shape: {self._X.shape}, y isn't provided\n")

        return trafo

    def run_after_splitting(
        self,
        X_tr: Union[pd.DataFrame, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        fold: int,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], object]:
        """Clean and process data after data splitting.

        To avoid data leakage, some data processing techniques should
        be applied after data is splitted.

        Parameters:
            X_tr: X training set
            X_val: X validation set
            fold: current fold number

        Return:
            X_tr: processed X training set
            X_val: processed X validation set
            scl: fittet scaler
        """
        logging.info("Run data cleaning and processing after data splitting...")
        scl = None
        if self.scale_cfg["type"] is not None:
            X_tr, X_val, scl = self._scale(X_tr, X_val)
            # =TODO=
            # Dump trafos and other objects elsewhere
            with open(os.path.join(DUMP_PATH, "trafos", f"fold{fold}.pkl"), "wb") as f:
                pickle.dump(scl, f)

        return X_tr, X_val, scl

    def get_df(self) -> pd.DataFrame:
        """Return raw DataFrame (i.e., initial snapshot)."""
        return self._df

    def get_X_y(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Return X set and y set."""
        return self._X, self._y

    def get_feats(self) -> List[str]:
        """Return features fed into the model."""
        return self._feats_slc

    # After data splitting
    def _scale(
        self,
        X_tr: Union[pd.DataFrame, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Any]:
        """Scale numeric features.

        Support only pd.DataFrame now.

        Return:
            X_tr: scaled X training set
            X_val: scaled X validation set
            scl: fittet scaler
        """
        assert isinstance(X_tr, pd.DataFrame) and isinstance(X_val, pd.DataFrame)

        scl_type = self.scale_cfg["type"]
        cols_to_trafo = self.scale_cfg["cols"]

        if scl_type == "minmax":
            scl = MinMaxScaler()
        elif scl_type == "standard":
            scl = StandardScaler()
        elif scl_type == "quantile":
            n_quantiles = self.scale_cfg["n_quantiles"]
            scl = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=168,
            )

        #         if cols_to_trafo == []:
        #             cols_to_trafo = _get_numeric_cols(X_tr)

        logging.info(f"Start scaling features using {scl_type} trafo...\n" f"Feature list:\n{cols_to_trafo}")
        X_tr[cols_to_trafo] = scl.fit_transform(X_tr[cols_to_trafo])
        X_val[cols_to_trafo] = scl.transform(X_val[cols_to_trafo])
        logging.info("Done.")

        X_tr.fillna(0, inplace=True)
        X_val.fillna(0, inplace=True)

        return X_tr, X_val, scl
