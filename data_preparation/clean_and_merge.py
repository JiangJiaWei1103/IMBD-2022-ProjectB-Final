"""Clean and merge sub-datasets into one complete dataset."""
import gc
import os
from argparse import Namespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from engine.defaults import BaseArgParser
from metadata import N_PROC_LAYERS, SG_FEAT, SPIKE_FEAT
from paths import PROC_DATA_PATH, RAW_DATA_PATH

DOWNCAST = True


class DCArgParser(BaseArgParser):
    """Argument parser for data cleaning."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--dataset", type=str, choices=["train1", "train2", "test"], default=None, help="name of the dataset"
        )


def _clean_and_merge(dataset_path: str, info_type: str, n_proc_layers: int) -> pd.DataFrame:
    """Clean and merge raw data.

    Parameters:
        dataset_path: path of the dataset
        info_type: information type, either sg or spike
        n_proc_layers: number of processing layers

    Return:
        data: merged data
    """
    raw_feats = SG_FEAT if info_type == "sg" else SPIKE_FEAT

    print(f"Start cleaning and merging {info_type} in {dataset_path}...")
    data = pd.DataFrame()
    for layer in tqdm(range(1, n_proc_layers + 1)):
        data_path = os.path.join(dataset_path, f"{layer}_{info_type}.csv")
        data_layer = pd.read_csv(data_path)
        data_layer["layer"] = layer
        data_layer.columns = ["time"] + raw_feats + ["layer"]

        assert data_layer["time"].is_monotonic_increasing, f"Time identifiers of {data_path} have some issues..."
        data = pd.concat([data, data_layer])
    print("Success.")

    if DOWNCAST:
        # Downcast feature columns to lower precision
        data[raw_feats] = data[raw_feats].astype(np.float32)

    return data


def main(args: Namespace) -> None:
    # Parse arguments
    dataset = args.dataset

    # Clean and merge raw data
    dataset_path = os.path.join(RAW_DATA_PATH, dataset)
    n_proc_layers = N_PROC_LAYERS[dataset]
    for info_type in ["sg", "spike"]:
        data = _clean_and_merge(dataset_path, info_type, n_proc_layers)

        # Dump datasets
        dump_path = os.path.join(PROC_DATA_PATH, f"{info_type}_{dataset}.csv")
        data.to_csv(dump_path, index=False)

        # Free mem.
        del data
        gc.collect()


if __name__ == "__main__":
    args = DCArgParser().parse()
    main(args)
