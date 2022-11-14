"""
Main script for training and evaluation processes using traditional ML.

This script supports following training and evaluation processes:
1. (`data_type`="aug", `mix_aug`=False)
*Note: This script is used to do chunk-aware modeling only.
"""
import logging
import os
import warnings
from argparse import Namespace
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

from data.data_processor import DataProcessor
from engine.defaults import TrainEvalArgParser
from experiment.experiment import Experiment
from metadata import GP1_N_CHUNKS, GP2_N_CHUNKS
from utils.evaluating import Evaluator
from validation.cross_validate import MultiSeedCVWrapper

warnings.simplefilter("ignore")

# Remove local plotting function when uploaded to IMBD remote
LOCAL = True
if LOCAL:
    from utils.eda import plot_pred_and_gt


def _get_data_chunk(dp: DataProcessor) -> Iterator[Tuple[int, int, pd.DataFrame, np.ndarray]]:
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
            logging.info(f">>>>>>> Chunk-Aware Modeling Group{gp_id} Chunk{chunk_id} <<<<<<<")
            dp.run_before_cv(gp_id=gp_id, chunk_id=chunk_id)
            X_chunk, y_chunk = dp.get_X_y()

            yield gp_id, chunk_id, X_chunk, y_chunk


def main(args: Namespace) -> None:
    """Run chunk-aware training and evaluation.

    Parameters:
        args: arguments driving training and evaluation processes

    Return:
        None
    """
    # Check args validity
    assert args.data_type == "aug", "You must use augmented data for chunk-aware modeling."
    assert args.mix_aug is False, "You must set `mix_aug`=False for chunk-aware modeling."

    # Configure experiment
    experiment = Experiment(args)

    with experiment as exp:
        exp.dump_cfg(exp.cfg, "cfg")

        # Prepare data
        dp = DataProcessor(dataset=args.dataset, data_type="aug", mix_aug=False, infer=False, **exp.dp_cfg)

        # Run chunk-aware CV
        oof_pred_final: Dict[str, Dict[str, np.ndarray]] = {"gp1": {}, "gp2": {}}
        for gp_id, chunk_id, X_chunk, y_chunk in _get_data_chunk(dp):
            cv_wrapper = MultiSeedCVWrapper(n_seeds=args.n_seeds, n_folds=args.n_folds, verbose=True, mix_aug=False)
            evaluator = Evaluator(args.dataset, eval_range=[f"gp{gp_id}"])
            oof_preds, *_, models, feat_imps = cv_wrapper.run_cv(
                X_chunk,
                y_chunk,
                model_name=args.model_name,
                model_params=exp.model_params,
                fit_params=exp.fit_params,
                evaluator=evaluator,
                group=None,
            )

            # Record average oof_pred over folds
            oof_pred_final[f"gp{gp_id}"][f"pred_with_chunk{chunk_id}"] = np.mean(oof_preds, axis=0)

            # Dump CV results
            obj_suffix = f"g{gp_id}_c{chunk_id}"
            exp.dump_ndarr(np.stack(oof_preds), f"oof/{obj_suffix}")
            if len(feat_imps) > 0:
                exp.dump_df(feat_imps, f"imp/feat_imps_{obj_suffix}", ext="csv")

            for sid, models_seed in enumerate(models):
                for mid, model in enumerate(models_seed):
                    exp.dump_model(model, model_type=obj_suffix, mid=f"seed{sid}_fold{mid}")

            with open(f"{exp.exp_dump_path}/feats/{obj_suffix}.txt", "w") as f:
                for feat in dp.get_feats():
                    f.write(f"{feat}\n")

        if LOCAL:
            y_base = pd.read_csv("./data/raw/train1/wear.csv")["MaxWear"].values
            plot_pred_and_gt(
                y_base[:26],
                oof_pred_final["gp1"],
                figsize=(12, 6),
                legend=True,
                dump_path=os.path.join(exp.exp_dump_path, "media", "gp1.jpg"),
            )
            plot_pred_and_gt(
                y_base[-20:],
                oof_pred_final["gp2"],
                figsize=(12, 6),
                legend=True,
                dump_path=os.path.join(exp.exp_dump_path, "media", "gp2.jpg"),
            )

            oof_pred_gp1_avg = np.mean(list(oof_pred_final["gp1"].values())[:-1], axis=0)  # Over chunks
            oof_pred_gp2_avg = np.mean(list(oof_pred_final["gp2"].values())[:-1], axis=0)  # Over chunks
            oof_pred_cat = np.concatenate((oof_pred_gp1_avg, oof_pred_gp2_avg))
            plot_pred_and_gt(
                y_base,
                [oof_pred_cat],
                figsize=(12, 6),
                legend=False,
                dump_path=os.path.join(exp.exp_dump_path, "media", "final.jpg"),
            )

        logging.info("\n≡≡≡≡≡≡≡ Final Performance Summary ≡≡≡≡≡≡≡")
        logging.info("==> Final score <==")
        final_evaluator = Evaluator(args.dataset)
        for eval_range, prf in final_evaluator.evaluate([oof_pred_cat]).items():
            logging.info(f"{eval_range.upper()}: {prf[0]:.5f} ± {prf[1]:.5f}")


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
