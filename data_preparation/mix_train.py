"""Mix train1 and train2."""
import os

import pandas as pd

from paths import PROC_DATA_PATH, RAW_DATA_PATH

TRAIN1_N_LAYERS = 46
GT_FILE_NAME = "00_Wear_data.csv"


def main() -> None:
    # Load data
    X_train1 = pd.read_csv(os.path.join(PROC_DATA_PATH, "X_train1.csv"))
    X_train2 = pd.read_csv(os.path.join(PROC_DATA_PATH, "X_train2.csv"))
    X_aug_train1 = pd.read_csv(os.path.join(PROC_DATA_PATH, "X_aug_train1.csv"))
    X_aug_train2 = pd.read_csv(os.path.join(PROC_DATA_PATH, "X_aug_train2.csv"))
    gt1 = pd.read_csv(os.path.join(RAW_DATA_PATH, "train1", GT_FILE_NAME))
    gt2 = pd.read_csv(os.path.join(RAW_DATA_PATH, "train2", GT_FILE_NAME))

    # Create pseudo layer identifiers
    X_train2["layer"] = X_train2["layer"] + TRAIN1_N_LAYERS
    X_aug_train2["layer"] = X_aug_train2["layer"] + TRAIN1_N_LAYERS
    gt2["Index"] = gt2["Index"] + TRAIN1_N_LAYERS

    # Mix train1 and train2 by concatenation
    X_mix = pd.concat([X_train1, X_train2], ignore_index=True)
    X_aug_mix = pd.concat([X_aug_train1, X_aug_train2], ignore_index=True)
    gt_mix = pd.concat([gt1, gt2], ignore_index=True)

    # Dump mixed data
    X_mix.to_csv(os.path.join(PROC_DATA_PATH, "X_mix.csv"), index=False)
    X_aug_mix.to_csv(os.path.join(PROC_DATA_PATH, "X_aug_mix.csv"), index=False)
    gt_mix.to_csv(os.path.join(RAW_DATA_PATH, "mix", GT_FILE_NAME), index=False)


if __name__ == "__main__":
    main()
