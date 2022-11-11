"""Feature engineer."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# from paths import OOF_META_FEATS_PATH, TEST_META_FEATS_PATH

PERCENTILES = [1, 5, 10, 20, 25, 30, 40, 60, 70, 75, 80, 90, 95, 99]  # 50 is median


def cal_simple_stats(
    data: pd.DataFrame,
    cast: bool = True,
) -> pd.Series:
    """Calculate and return simple stats of the given data.

    Parameters:
        data: feature values
            *Note: Data can be raw data or pre-aggregated one (e.g.,
                rolling, window, specified chunk).
        downcast: whether to downcast to `np.float32`

    Return:
        feats: stats of the given data
    """
    feats = {}

    # Naive stats
    feats["min"] = data.min()
    feats["max"] = data.max()
    feats["mean"] = data.mean()
    feats["median"] = data.median()
    feats["std"] = data.std()
    feats["skew"] = data.skew()
    feats["kurtosis"] = data.kurt()
    feats["mad"] = data.mad()

    # Absolute stats
    feats["abs_min"] = data.abs().min()
    feats["abs_max"] = data.abs().max()
    feats["abs_mean"] = data.abs().mean()
    feats["abs_median"] = data.abs().median()
    feats["abs_std"] = data.abs().std()

    # Quantile
    for p in PERCENTILES:
        feats[f"p{p}"] = data.quantile(p / 100)
        feats[f"abs_p{p}"] = data.abs().quantile(p / 100)
    feats["iqr"] = data.quantile(0.75) - data.quantile(0.25)

    # Threshold count
    #     feats[f"thres{n}_cnt"] = (data.abs() > XX).sum()

    # Compound
    feats["max_to_min"] = feats["max"] / feats["abs_min"]
    feats["max_to_min_diff"] = feats["max"] - feats["abs_min"]

    feats = _dict2series(feats)
    if cast:
        feats = feats.astype(np.float32)  # type: ignore

    return feats


def cal_trends(
    data: pd.DataFrame,
    abs_val: bool = False,
    cast: bool = True,
) -> pd.Series:
    """Calculate and return trend features of the given data.

    Parameters:
        data: feature values
            *Note: Data can be raw data or pre-aggregated one (e.g.,
                rolling, window, specified chunk).
        abs_val: whether to calculate trend features based on absolute
            values
        downcast: whether to downcast to `np.float32`

    Return:
        feats: trend features of the given data
    """
    feats = {}

    if abs_val:
        data = data.abs()

    for col in data.columns:
        y = data[col].dropna().reset_index(drop=True)
        x_idx = np.arange(len(y)).reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(x_idx, y)
        feats[f"{col}_coef"] = lr.coef_[0]
        feats[f"{col}_intercept_"] = lr.intercept_

    feats = pd.Series(feats)
    if cast:
        feats = feats.astype(np.float32)  # type: ignore

    return feats


def cal_sta_lta(
    data: pd.DataFrame,
    win_lens: Tuple[int, int],
    cast: bool = True,
) -> pd.Series:
    """Calculate and return STA/LTA of the given data.

    Parameters:
        data: feature values
            *Note: Data can be raw data or pre-aggregated one (e.g.,
                rolling, window, specified chunk).
        win_lens: short-term and long-term window lengths
            *Note: The format should be (`sta_len`, `lta_len`)
        downcast: whether to downcast to `np.float32`

    Return:
        feats: STA/LTA of the given data
    """
    feats = {}

    sta_len, lta_len = win_lens

    def _cal_sta_lta_col(col: str) -> np.ndarray:
        sta = np.cumsum(data[col] ** 2).array
        lta = sta.copy()

        sta[sta_len:] = sta[sta_len:] - sta[:-sta_len]
        sta = sta / sta_len
        lta[lta_len:] = lta[lta_len:] - lta[:-lta_len]
        lta = lta / lta_len

        sta[: lta_len - 1] = 0

        lta_tiny = np.finfo(0.0).tiny
        lta_tiny_mask = lta < lta_tiny
        lta[lta_tiny_mask] = lta_tiny

        return sta / lta

    for col in data.columns:
        feats[f"{col}_sta{sta_len}lta{lta_len}_mean"] = _cal_sta_lta_col(col).mean()
        feats[f"{col}_sta{sta_len}lta{lta_len}_std"] = _cal_sta_lta_col(col).std()

    feats = pd.Series(feats)
    if cast:
        feats = feats.astype(np.float32)  # type: ignore

    return feats


def cal_val_changes(
    data: pd.DataFrame,
    cast: bool = True,
) -> pd.DataFrame:
    """Calculate and return feature value changes along sequence.

    Parameters:
        data: feature values
            *Note: Data can be raw data or pre-aggregated one (e.g.,
                rolling, window, specified chunk).
        downcast: whether to downcast to `np.float32`

    Return:
        feats: feature value changes of the given data
    """
    feats = {}

    # Difference
    data_diff = data.diff().drop(0, axis=0).reset_index(drop=True)
    data_diff.columns = [f"{col}_diff" for col in data.columns]
    data_diff_stats = cal_simple_stats(data_diff)
    data_diff_trends = cal_trends(data_diff)
    data_diff_stalta = cal_sta_lta(data_diff, (500, 10000))  # Tmp hardcoded

    # Rate of change
    data_roc = data_diff / data[:-1].values
    data_roc = _mask_inf(data_roc)
    data_roc.columns = [f"{col}_roc" for col in data.columns]
    data_roc_stats = cal_simple_stats(data_roc)
    data_roc_trends = cal_trends(data_roc)
    data_roc_stalta = cal_sta_lta(data_roc, (500, 10000))  # Tmp hardcoded

    feats = pd.concat(
        [data_diff_stats, data_diff_trends, data_diff_stalta, data_roc_stats, data_roc_trends, data_roc_stalta]
    )
    if cast:
        feats = feats.astype(np.float32)  # type: ignore

    return feats


def _dict2series(feats: Dict[str, pd.Series]) -> pd.Series:
    """Convert newly engineered features to `pd.Series` representation.

    Parameters:
        feats: newly engineered features stored in dictionary

    Return:
        feats_series: newly engineered features stored in series
    """
    # Construct new feature names
    base_feats = list(feats.values())[0].index.tolist()
    new_feat_suffix = list(feats.keys())
    new_feats = [f"{base_feat}_{s}" for s in new_feat_suffix for base_feat in base_feats]

    # Construct feature series
    feats_series = np.hstack(feats.values())  # type: ignore
    feats_series = pd.Series(feats_series, index=new_feats)

    return feats_series


def _mask_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Use NaN to mask +-inf."""
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# 以下不用看XD
class FE:
    """Feature engineer.

    Parameters:
        ...

        infer: whether the process is in inference mode
    """

    MV2EID = {
        "l0": "lgbm-xxxxxxxx",
    }  # Base model version to corresponding experiment identifier
    EPS: float = 1e-7
    _df: pd.DataFrame = None  # FE is applied to this DataFrame (main obj for data flowing)
    _eng_feats: List[str] = []  # Newly engineered features
    _cat_feats: List[str] = []  # Newly engineered categorical features (how about the original ones?)

    def __init__(
        self,
        #         add_inter_subproc_diff: bool,   # Example acting as placeholder
        infer: bool = False,
    ):
        #         self.add_inter_subproc_diff = add_inter_subproc_diff

        self.infer = infer

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering.

        Parameters:
            df: input DataFrame

        Return:
            self._df: DataFrame with engineered features
        """
        self._df = df.copy()

        #         if self.add_inter_subproc_diff:
        #             self._add_inter...()

        return self._df

    def get_eng_feats(self) -> List[str]:
        """Return list of all engineered features."""
        return self._eng_feats

    def get_cat_feats(self) -> List[str]:
        """Return list of categorical features."""
        return self._cat_feats

    """Meta features are used for stacking, so leave it here?
    def _add_meta_feats(self) -> None:
        '''Add meta features for stacking or restacking.'''
        if self.infer:
            # Testing prediction is used
            meta_feats = pd.read_csv(TEST_META_FEATS_PATH)
        else:
            # Unseen prediction is used
            meta_feats = pd.read_csv(OOF_META_FEATS_PATH)

        print("Adding meta features...")
        meta_cols = []
        for model_v in self.meta_feats:
            meta_cols.append(self.MV2EID[model_v])
        meta_feats = meta_feats[PK + meta_cols]

        #         for meta_col in meta_cols:
        #             meta_feats[meta_col] = (meta_feats[meta_col]
        #                                     / meta_feats["Capacity"])

        self._df = self._df.merge(meta_feats, how="left", on=PK, validate="1:1")
        print("Done.")

        self._eng_feats += meta_cols

    def _add_knn_meta_feats(self) -> None:
        '''Add meta features from kNN.

        Illustration of kNN meta column conversion:
            {
                "l5": 2,
                "l6": 3,
                ...
                <model version>: k
            }
            -> {
                "lgbm-hjc3rp0j": 2,
                "lgbm-54or6r30": 3,
                ...
                <experiment identifier>: k
            }
        '''
        if self.infer:
            # Testing prediction is used
            meta_feats = pd.read_csv(TEST_META_FEATS_PATH)
        else:
            # Unseen prediction is used
            meta_feats = pd.read_csv(OOF_META_FEATS_PATH)

        # Load geographic kNN of each generator
        with open("./data/processed/gen_geo_knn.pkl", "rb") as f:
            geo_knn = pickle.load(f)

        print("Adding kNN meta features...")
        for model_v, k in self.knn_meta_feats.items():
            self.knn_meta_feats[self.MV2EID[model_v]] = self.knn_meta_feats.pop(model_v)

        knn_meta_cols = [
            f"{meta_col}_n{i}"
            for meta_col, k in self.knn_meta_feats.items()
            for i in range(k)
        ]
        knn_meta_dict: Dict[str, List[float]] = {col: [] for col in knn_meta_cols}
        for i, r in self._df.iterrows():
            meta_feats_date = meta_feats[meta_feats["Date"] == r["Date"]]
            cap = str(r["Capacity"])

            for meta_col, k in self.knn_meta_feats.items():
                knn = geo_knn[cap][:k]

                for i, cap_ in enumerate(knn):
                    cap_ = float(cap_)
                    df_knn = meta_feats_date[meta_feats_date["Capacity"] == cap_]

                    knn_meta_col = f"{meta_col}_n{i}"
                    if len(df_knn) != 0:
                        knn_meta_dict[knn_meta_col].append(
                            df_knn[meta_col].values[0]  # / cap_
                        )
                    else:
                        knn_meta_dict[knn_meta_col].append(np.nan)

        knn_meta_df = pd.DataFrame.from_dict(knn_meta_dict)
        self._df = pd.concat([self._df, knn_meta_df], axis=1)
        print("Done.")

        self._eng_feats += knn_meta_cols

    """
