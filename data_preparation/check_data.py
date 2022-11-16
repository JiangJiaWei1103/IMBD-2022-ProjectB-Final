"""Perform data checking."""
import logging
import os
from argparse import Namespace

import numpy as np
import pandas as pd

from engine.defaults import BaseArgParser
from metadata import PK
from paths import PROC_DATA_PATH
from utils.eda import summarize
from utils.logger import Logger

FREQ_RATIO = 2.5  # Sampling frequency ratio (spike / sg)


class DateCheckArgParser(BaseArgParser):
    """Argument parser for data augmentation."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--dataset", type=str, choices=["train1", "train2", "test"], default=None, help="name of the dataset"
        )


def main(args: Namespace) -> None:
    """Perform data checking.

    Parameters:
        args: arguments driving data checking process

    Return:
        None
    """
    dataset = args.dataset
    _ = Logger(logging_file=f"./eda_result/data_check_{dataset}.log").get_logger()
    logging.info(f"=====Data Check on {dataset}=====\n")

    # Load data
    logging.info(f"Load sg_{dataset} and spike_{dataset}...")
    sg = pd.read_csv(os.path.join(PROC_DATA_PATH, f"sg_{dataset}.csv"))
    spike = pd.read_csv(os.path.join(PROC_DATA_PATH, f"spike_{dataset}.csv"))

    # Summarize data
    logging.info("Check data shape, zero ratios and NaN raitos...")
    summarize(sg, "SG", 2)
    summarize(spike, "SPIKE", 2)

    # Check if spike is roughly 2.5 more samples than sg
    logging.info(f"\nCheck #Sample of spike / #Samples of sg -> {FREQ_RATIO} or not...")
    logging.info(f"#Samples of spike / #Samples of sg: {len(spike) / len(sg)}")

    # Check number of samples per layer
    logging.info("\nCheck if sg and spike information are sliced into two groups (like train1 26 / 20)...")
    logging.info("(Directly check groups' boundary (min/max) and #layers per group (count).)")
    n_samples_per_layer_sg = sg.groupby(PK)["time"].count().to_frame(name="n_samples")
    n_samples_per_layer_spike = spike.groupby(PK)["time"].count().to_frame(name="n_samples")
    logging.info("=====sg=====")
    logging.info(n_samples_per_layer_sg.T)
    logging.info(n_samples_per_layer_sg.reset_index().groupby("n_samples")["layer"].agg(["min", "max", "count"]))
    logging.info("=====spike=====")
    logging.info(n_samples_per_layer_spike.T)
    logging.info(n_samples_per_layer_spike.reset_index().groupby("n_samples")["layer"].agg(["min", "max", "count"]))

    # Check negative threshold of sg["e"].diff() to facilitate DA slicing
    logging.info('\nCheck sg["e"].diff().sort_values() to help DA slicing...')
    for layer, gp in sg.groupby(PK):
        e_diff = gp["e"].diff().sort_values().reset_index(drop=True)
        e_diff_ = np.round_(e_diff[:30].values, 2)
        e_diff_pret = []
        for i, val in enumerate(e_diff_):
            val_suffix = " "
            if (i + 1) % 10 == 0:
                val_suffix = "\n" + " " * 10
            e_diff_pret.append(str(val) + val_suffix)
        e_diff_pret = "".join(e_diff_pret)
        logging.info(f"Layer{layer: >3}: {e_diff_pret}")


if __name__ == "__main__":
    args = DateCheckArgParser().parse()
    main(args)
