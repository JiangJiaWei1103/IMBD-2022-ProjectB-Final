"""Slice intra-layer feature sequence to fine-grained operations."""
import os
from argparse import Namespace
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from engine.defaults import BaseArgParser
from paths import PROC_DATA_PATH

FREQ_RATIO = 2.5  # Sampling frequency ratio (spike / sg)


class DAArgParser(BaseArgParser):
    """Argument parser for data augmentation."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--dataset", type=str, choices=["train1", "train2", "test"], default=None, help="name of the dataset"
        )
        self.argparser.add_argument(
            "--neg-peak-thres",
            type=float,
            default=None,
            help="negative peak thres of `e` diff in SG used to flag slice points",
        )


def _get_slice_pts(
    sg: pd.DataFrame,
    neg_peak_thres: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Flag and return slicing points of sg and spike info.

    Parameters:
        sg: raw sg information
        neg_peak_thres: negative peak threshold of `e` difference

    Return:
        slice_pts_sg: slicing points of sg info
        slice_pts_spike: slicing points for spike info
    """
    sg_gp = sg.groupby("layer")

    # Flag slicing points
    slice_pts_sg = {}
    for layer, gp in sg_gp:
        # Don't reset `gp` index, or slicing points can't match original indices
        gp_e_diff = gp["e"].diff()
        slice_pts_sg_single_layer = gp_e_diff.index[gp_e_diff < neg_peak_thres]
        slice_pts_sg[layer] = slice_pts_sg_single_layer
    slice_pts_spike = {layer: np.round(v * FREQ_RATIO).astype(int) for layer, v in slice_pts_sg.items()}

    return slice_pts_sg, slice_pts_spike


def _aug_by_slicing(
    data: pd.DataFrame,
    slice_pts: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """Run DA by slicing each processing layer to fine-grained chunks.

    Parameters:
        data: raw data, either sg or spike information
        slice_pts: slicing points

    Return:
        data_aug: augmented data
    """

    def _fill_logic(x: pd.Series) -> pd.Series:
        return x.bfill().fillna(x.max() + 1)

    data_aug = data.copy()
    slice_pt_seq = np.hstack(list(slice_pts.values()))
    data_aug["slice"] = np.nan
    data_aug.loc[slice_pt_seq, "slice"] = np.hstack([np.arange(len(v)) for v in slice_pts.values()])
    data_aug["slice"] = data_aug.groupby("layer")["slice"].apply(_fill_logic).astype(np.int32)

    return data_aug


def main(args: Namespace) -> None:
    # Retrieve arguments
    dataset = args.dataset
    neg_peak_thres = args.neg_peak_thres

    # Load processed (merged) data
    sg_file_path = os.path.join(PROC_DATA_PATH, f"sg_{dataset}.csv")
    spike_file_path = os.path.join(PROC_DATA_PATH, f"spike_{dataset}.csv")
    sg = pd.read_csv(sg_file_path)
    spike = pd.read_csv(spike_file_path)

    # Flag slicing points
    slice_pts_sg, slice_pts_spike = _get_slice_pts(sg, neg_peak_thres)

    # Slice layer-wise data to fine-grained chunks
    sg_aug = _aug_by_slicing(sg, slice_pts_sg)
    spike_aug = _aug_by_slicing(spike, slice_pts_spike)

    # Dump augmented datasets
    sg_aug.to_csv(os.path.join(PROC_DATA_PATH, f"sg_aug_{dataset}.csv"), index=False)
    spike_aug.to_csv(os.path.join(PROC_DATA_PATH, f"spike_aug_{dataset}.csv"), index=False)


if __name__ == "__main__":
    args = DAArgParser().parse()
    main(args)
