"""Generate hand-crafted features."""
import os
from argparse import Namespace
from typing import Any, Tuple, Union

import pandas as pd
from joblib import Parallel, delayed

from data.fe import cal_simple_stats, cal_sta_lta, cal_trends, cal_val_changes
from engine.defaults import BaseArgParser
from metadata import PK, PK_AUG, SG_FEAT, SPIKE_FEAT
from paths import PROC_DATA_PATH


class FEArgParser(BaseArgParser):
    """Argument parser for feature engineering."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--dataset", type=str, choices=["train1", "train2", "test"], default=None, help="name of the dataset"
        )
        self.argparser.add_argument(
            "--data-type", type=str, choices=["normal", "aug"], default=None, help="type of data"
        )


def _run_chunked_fe(
    chunk_id: Union[int, Tuple[int, int]], data: pd.DataFrame, info_type: str, **kwargs: Any
) -> pd.Series:
    """Run feature engineering to get the complete feature set for a
    single data chunk.

    Parameters:
        chunk_id: chunk identifier, `layer` or (`layer`, `slice`) pair
        data: feature values
            *Note: Data can be raw data or pre-aggregated one (e.g.,
                rolling, window, specified chunk).
        info_type: type of information, either `sg` or `spike`

    Return:
        feats: feature set for a single data chunk
    """
    raw_feats = SG_FEAT if info_type == "sg" else SPIKE_FEAT
    data = data.reset_index(drop=True)[raw_feats]
    data.columns = [f"{info_type}_{col}" for col in data.columns]

    # Run feature engineering pipeline to get complete feature set
    stats = cal_simple_stats(data)
    trends = cal_trends(data)
    stalta = cal_sta_lta(data, (5000, 50000))
    val_changes = cal_val_changes(data)

    feats = pd.concat([pd.Series(chunk_id), stats, trends, stalta, val_changes])

    # Rename PK columns
    if isinstance(chunk_id, int):
        feats.rename({0: "layer"}, inplace=True)
    else:
        feats.rename({0: "layer", 1: "slice"}, inplace=True)

    return feats


def main(args: Namespace) -> None:
    # Retrieve arguments
    dataset = args.dataset
    data_type = args.data_type

    if data_type == "aug":
        data_type = "aug_"
        data_pk = PK_AUG
    else:
        data_type = ""
        data_pk = PK

    # Load data
    sg_file_path = os.path.join(PROC_DATA_PATH, f"sg_{data_type}{dataset}.csv")
    spike_file_path = os.path.join(PROC_DATA_PATH, f"spike_{data_type}{dataset}.csv")
    sg = pd.read_csv(sg_file_path)
    spike = pd.read_csv(spike_file_path)

    # Run FE
    sg_feats = Parallel(n_jobs=12, backend="threading", verbose=10)(
        delayed(_run_chunked_fe)(chunk_id, data, "sg") for chunk_id, data in sg.groupby(data_pk)
    )
    spike_feats = Parallel(n_jobs=12, backend="threading", verbose=10)(
        delayed(_run_chunked_fe)(chunk_id, data, "spike") for chunk_id, data in spike.groupby(data_pk)
    )
    df_sg = pd.DataFrame(sg_feats)
    df_spike = pd.DataFrame(spike_feats)
    df = df_sg.merge(df_spike, on=data_pk)

    # Dump augmented datasets
    df.to_csv(os.path.join(PROC_DATA_PATH, f"X_{data_type}{dataset}.csv"), index=False)


if __name__ == "__main__":
    args = FEArgParser().parse()
    main(args)
