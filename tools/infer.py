"""Main script for inference.

[ ] Plot if LOCAL, think a way to visual if not LOCAL
[ ] Then compare, interpret results and use another script to try final ensemble
"""
import logging
import os
import pickle
import sys
from argparse import Namespace
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

from data.data_processor import DataProcessor
from engine.defaults import InferArgParser
from experiment.experiment import Experiment
from metadata import GP1_N_CHUNKS, GP2_N_CHUNKS
from utils.aug_pred_blender import AugPredBlender
from utils.evaluating import Evaluator

# Remove local plotting function when uploaded to IMBD remote
LOCAL = True
if LOCAL:
    from utils.eda import plot_pred_and_gt


def _parse_modeling_strategy(modeling_strategy: str) -> Tuple[str, bool]:
    """Parse modeling strategy of pre-trained models.

    Parameters:
        modeling_strategy: modeling strategy, the choices are:
            {"normal", "aug", "chunk"}

    Return:
        data_type: type of the data, the choices are as follows:
            {"normal", "aug"}
        mix_aug: if True, models are trained on data chunks mixed
            together
    """
    mix_aug = False
    if modeling_strategy == "normal":
        data_type = "normal"
    elif modeling_strategy == "chunk":
        data_type = "aug"
    elif modeling_strategy == "aug":
        data_type = "aug"
        mix_aug = True
    else:
        print(f"Modeling strategy {modeling_strategy} isn't registered.")
        sys.exit(1)

    return data_type, mix_aug


def _get_data_chunk(dp: DataProcessor, exp_dump_path: str) -> Iterator[Tuple[int, int, pd.DataFrame, np.ndarray]]:
    """Retrieve and return data chunk and corresponding identifiers.

    Yield:
        gp_id: group identifier, either 1 or 2
        chunk_id: chunk (slice) identifier
        X_chunk: chunked X set
        y_chunk: chunk y set
    """
    for gp_id in [1, 2]:
        n_chunks = GP1_N_CHUNKS[dp.dataset] if gp_id == 1 else GP2_N_CHUNKS[dp.dataset]
        for chunk_id in range(0, n_chunks):
            logging.info(f">>>>>>> Chunk-Aware Inference Group{gp_id} Chunk{chunk_id} <<<<<<<")
            obj_suffix = f"g{gp_id}_c{chunk_id}"
            feats = []
            with open(os.path.join(exp_dump_path, "feats", f"{obj_suffix}.txt"), "r") as f:
                for feat in f.readlines():
                    feats.append(feat.strip())
            with open(os.path.join(exp_dump_path, "trafos", f"{obj_suffix}.pkl"), "rb") as f:
                trafo = pickle.load(f)
            _ = dp.run_before_cv(feats_to_use=feats, trafo=trafo, gp_id=gp_id, chunk_id=chunk_id)
            X_chunk, y_chunk = dp.get_X_y()

            yield gp_id, chunk_id, X_chunk, y_chunk


def _run_infer(X: pd.DataFrame, model_dir: str, n_seeds: int, n_folds: int) -> np.ndarray:
    """Run inference and return predicting results.

    For a single inference process, predictions are averaged over all
    models obtained in single CV round (i.e., #models=n_seeds*n_folds).

    Parameters:
        X: features to feed into pre-trained models
        model_dir: directory of dumped pre-trained models
        n_seeds: number of seeds used in a single CV round
        n_folds: number of folds in a single data splitting

    Return:
        y_pred: average predicting result
    """
    n_models = n_seeds * n_folds
    y_pred = np.zeros(len(X))
    for sid in range(n_seeds):
        for fid in range(n_folds):
            model_path = os.path.join(model_dir, f"seed{sid}_fold{fid}.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            y_pred += model.predict(X) / n_models

    return y_pred


def main(args: Namespace) -> None:
    """Run chunk-aware training and evaluation.

    Parameters:
        args: arguments driving training and evaluation processes

    Return:
        None
    """
    assert args.exp_id is not None, "You must specify the version of pre-trained models to use."
    if args.dataset == "test":
        assert args.eval_holdout is False, "For test, you don't have groundtruths to do evaluation."

    # Parse experiment identifier
    _, modeling_strategy, _ = args.exp_id.split("-")
    data_type, mix_aug = _parse_modeling_strategy(modeling_strategy)

    # Configure experiment
    infer_target = "holdout" if args.eval_holdout else "final"
    experiment = Experiment(args, infer=True, log_file=f"infer_{infer_target}.log")

    with experiment as exp:
        # Prepare data
        dp = DataProcessor(dataset=args.dataset, data_type=data_type, mix_aug=mix_aug, infer=True, **exp.cfg["dp"])

        if modeling_strategy != "chunk":
            # Prepare X and y (optional) sets
            feats = []
            with open(os.path.join(exp.exp_dump_path, "feats.txt"), "r") as f:
                for feat in f.readlines():
                    feats.append(feat.strip())
            with open(os.path.join(exp.exp_dump_path, "trafos", "trafo.pkl"), "rb") as f:
                trafo = pickle.load(f)
            _ = dp.run_before_cv(feats_to_use=feats, trafo=trafo)
            X, y = dp.get_X_y()
            y_pred = _run_infer(
                X,
                os.path.join(exp.exp_dump_path, "models"),
                exp.cfg["common"]["n_seeds"],
                exp.cfg["common"]["n_folds"],
            )
            if data_type == "aug":
                y_pred = AugPredBlender().blend(dp.get_df()["layer"], y_pred)

            exp.dump_ndarr(y_pred, infer_target)
            if LOCAL:
                y_base = pd.read_csv("./data/raw/train1/wear.csv")["MaxWear"].values
                plot_pred_and_gt(
                    y_base,
                    [y_pred],
                    figsize=(12, 6),
                    legend=True,
                    dump_path=os.path.join(exp.exp_dump_path, f"media/{infer_target}", f"{infer_target}.jpg"),
                )
        else:
            # Run chunk-aware inference
            y_pred_gp: Dict[str, Dict[str, np.ndarray]] = {"gp1": {}, "gp2": {}}
            for gp_id, chunk_id, X_chunk, y_chunk in _get_data_chunk(dp, exp.exp_dump_path):
                obj_suffix = f"g{gp_id}_c{chunk_id}"
                model_dir = os.path.join(exp.exp_dump_path, "models", obj_suffix)
                y_pred_chunk = _run_infer(
                    X_chunk, model_dir, exp.cfg["common"]["n_seeds"], exp.cfg["common"]["n_folds"]
                )
                y_pred_gp[f"gp{gp_id}"][f"pred_with_chunk{chunk_id}"] = y_pred_chunk

                # Dumpc chunk-wise prediction
                exp.dump_ndarr(y_pred_chunk, f"{infer_target}/{obj_suffix}")

            # Aggregate predictions over chunks
            y_pred_gp1_avg = np.mean(
                list(y_pred_gp["gp1"].values())[:-1], axis=0
            )  # Take -1 if last chunk is too short
            y_pred_gp2_avg = np.mean(
                list(y_pred_gp["gp2"].values())[:-1], axis=0
            )  # Take -1 if last chunk is too short
            y_pred = np.concatenate((y_pred_gp1_avg, y_pred_gp2_avg))
            exp.dump_ndarr(y_pred, infer_target)  # Facilitate direct submission or next-stage ensemble

            if LOCAL:
                y_base = pd.read_csv("./data/raw/train1/wear.csv")["MaxWear"].values
                plot_pred_and_gt(
                    y_base[:26],
                    y_pred_gp["gp1"],
                    figsize=(12, 6),
                    legend=True,
                    dump_path=os.path.join(exp.exp_dump_path, f"media/{infer_target}", "gp1.jpg"),
                )
                plot_pred_and_gt(
                    y_base[-20:],
                    y_pred_gp["gp2"],
                    figsize=(12, 6),
                    legend=True,
                    dump_path=os.path.join(exp.exp_dump_path, f"media/{infer_target}", "gp2.jpg"),
                )
                plot_pred_and_gt(
                    y_base,
                    [y_pred],
                    figsize=(12, 6),
                    legend=False,
                    dump_path=os.path.join(exp.exp_dump_path, f"media/{infer_target}", "cat.jpg"),
                )

        if args.eval_holdout:
            evaluator = Evaluator(args.dataset)
            final_score = evaluator.evaluate([y_pred])
            logging.info("≡≡≡≡≡≡≡ Overall Performance Summary ≡≡≡≡≡≡≡")
            logging.info("==> Final score <==")
            for eval_range, prf in final_score.items():
                logging.info(f"{eval_range.upper()}: {prf[0]:.5f} ± {prf[1]:.5f}")
            logging.info("-" * 50)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = InferArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
