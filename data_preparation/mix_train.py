"""Mix train1 and train2."""
import os

import numpy as np
import pandas as pd

from metadata import GP1_LEN, GP1_N_CHUNKS, GP2_LEN, GP2_N_CHUNKS
from paths import PROC_DATA_PATH, RAW_DATA_PATH

TRAIN1_N_LAYERS = 46
GT_FILE_NAME = "00_Wear_data.csv"


def main() -> None:
    gp1_len_tr1, gp1_len_tr2 = GP1_LEN["train1"], GP1_LEN["train2"]
    gp2_len_tr1, gp2_len_tr2 = GP2_LEN["train1"], GP2_LEN["train2"]
    gp1_n_chunks_tr1, gp2_n_chunks_tr1 = GP1_N_CHUNKS["train1"], GP2_N_CHUNKS["train1"]
    gp1_n_chunks_tr2, gp2_n_chunks_tr2 = GP1_N_CHUNKS["train2"], GP2_N_CHUNKS["train2"]

    # Mix normal train1 and train2
    X_train1 = pd.read_csv(os.path.join(PROC_DATA_PATH, "X_train1.csv"))
    X_train2 = pd.read_csv(os.path.join(PROC_DATA_PATH, "X_train2.csv"))
    X_train1_gp1, X_train1_gp2 = X_train1[X_train1["layer"] <= gp1_len_tr1], X_train1[X_train1["layer"] > gp1_len_tr1]
    X_train2_gp1, X_train2_gp2 = X_train2[X_train2["layer"] <= gp1_len_tr2], X_train2[X_train2["layer"] > gp1_len_tr2]
    X_mix = pd.concat([X_train1_gp1, X_train2_gp1, X_train1_gp2, X_train2_gp2], ignore_index=True)
    X_mix["layer"] = np.arange(len(X_mix)).astype(np.int32) + 1  # Recode layer id

    # Mix augmented train1 and train2
    X_aug_train1 = pd.read_csv(os.path.join(PROC_DATA_PATH, "X_aug_train1.csv"))
    X_aug_train2 = pd.read_csv(os.path.join(PROC_DATA_PATH, "X_aug_train2.csv"))
    X_aug_train1_gp1, X_aug_train1_gp2 = (
        X_aug_train1[X_aug_train1["layer"] <= gp1_len_tr1],
        X_aug_train1[X_aug_train1["layer"] > gp1_len_tr1],
    )
    X_aug_train2_gp1, X_aug_train2_gp2 = (
        X_aug_train2[X_aug_train2["layer"] <= gp1_len_tr2],
        X_aug_train2[X_aug_train2["layer"] > gp1_len_tr2],
    )
    X_aug_mix = pd.concat([X_aug_train1_gp1, X_aug_train2_gp1, X_aug_train1_gp2, X_aug_train2_gp2], ignore_index=True)
    # Recode layer id
    layer_id = []
    for pseudo_layer in X_mix["layer"].values:
        if pseudo_layer <= gp1_len_tr1:
            n_chunks = gp1_n_chunks_tr1
        elif pseudo_layer > gp1_len_tr1 and pseudo_layer <= gp1_len_tr1 + gp1_len_tr2:
            n_chunks = gp1_n_chunks_tr2
        elif pseudo_layer > gp1_len_tr1 + gp1_len_tr2 and pseudo_layer <= gp1_len_tr1 + gp1_len_tr2 + gp2_len_tr1:
            n_chunks = gp2_n_chunks_tr1
        else:
            n_chunks = gp2_n_chunks_tr2
        layer_id.extend([pseudo_layer] * n_chunks)
    X_aug_mix["layer"] = np.array(layer_id).astype(np.int32)

    # Mix groundtruths of train1 and train2
    gt1 = pd.read_csv(os.path.join(RAW_DATA_PATH, "train1", GT_FILE_NAME))
    gt2 = pd.read_csv(os.path.join(RAW_DATA_PATH, "train2", GT_FILE_NAME))
    gt1_gp1, gt1_gp2 = gt1[gt1["Index"] <= gp1_len_tr1], gt1[gt1["Index"] > gp1_len_tr1]
    gt2_gp1, gt2_gp2 = gt2[gt2["Index"] <= gp1_len_tr2], gt2[gt2["Index"] > gp1_len_tr2]
    gt_mix = pd.concat([gt1_gp1, gt2_gp1, gt1_gp2, gt2_gp2], ignore_index=True)
    gt_mix["Index"] = np.arange(len(gt_mix)).astype(np.int32) + 1  # Recode layer id

    # Dump mixed data
    X_mix.to_csv(os.path.join(PROC_DATA_PATH, "X_mix.csv"), index=False)
    X_aug_mix.to_csv(os.path.join(PROC_DATA_PATH, "X_aug_mix.csv"), index=False)
    gt_mix.to_csv(os.path.join(RAW_DATA_PATH, "mix", GT_FILE_NAME), index=False)


if __name__ == "__main__":
    main()
