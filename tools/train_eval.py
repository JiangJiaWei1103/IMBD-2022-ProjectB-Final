"""
Main script for training and evaluation processes using traditional ML.

This script supports following training and evaluation processes:
1. (`data_type`="normal", `mix_aug`=False)
2. (`data_type`="aug", `mix_aug`=True)
*Note: To use chunk-aware modeling, please use `train_eval_chunk.py`.
"""
import os
import warnings
from argparse import Namespace

import numpy as np
import pandas as pd

from data.data_processor import DataProcessor
from engine.defaults import TrainEvalArgParser
from experiment.experiment import Experiment
from utils.evaluating import Evaluator
from validation.cross_validate import MultiSeedCVWrapper

warnings.simplefilter("ignore")

# Remove local plotting function when uploaded to IMBD remote
LOCAL = True
if LOCAL:
    from utils.eda import plot_pred_and_gt


def main(args: Namespace) -> None:
    """Run training and evaluation.

    Parameters:
        args: arguments driving training and evaluation processes

    Return:
        None
    """
    # Configure experiment
    experiment = Experiment(args)

    with experiment as exp:
        exp.dump_cfg(exp.cfg, "cfg")

        # Prepare data
        dp = DataProcessor(
            dataset=args.dataset, data_type=args.data_type, mix_aug=args.mix_aug, infer=False, **exp.dp_cfg
        )
        dp.run_before_cv()
        group = dp.get_df()["layer"] if args.mix_aug else None
        X, y = dp.get_X_y()

        # Run CV
        cv_wrapper = MultiSeedCVWrapper(n_seeds=args.n_seeds, n_folds=args.n_folds, verbose=True, mix_aug=args.mix_aug)
        evaluator = Evaluator(args.dataset)
        oof_preds, *_, models, feat_imps = cv_wrapper.run_cv(
            X,
            y,
            model_name=args.model_name,
            model_params=exp.model_params,
            fit_params=exp.fit_params,
            evaluator=evaluator,
            group=group,
        )

        # Dump CV results
        exp.dump_ndarr(np.stack(oof_preds), "oof")
        exp.dump_df(feat_imps, "imp/feat_imps", ext="csv")

        for sid, models_seed in enumerate(models):
            for mid, model in enumerate(models_seed):
                exp.dump_model(model, model_type="", mid=f"seed{sid}_fold{mid}")

        with open(f"{exp.exp_dump_path}/feats.txt", "w") as f:
            for feat in dp.get_feats():
                f.write(f"{feat}\n")

        if LOCAL:
            y_base = pd.read_csv("./data/raw/train1/wear.csv")["MaxWear"].values
            plot_pred_and_gt(
                y_base,
                oof_preds,
                figsize=(12, 6),
                legend=False,
                dump_path=os.path.join(exp.exp_dump_path, "media", "seed_by_seed.jpg"),
            )
            plot_pred_and_gt(
                y_base,
                [np.mean(oof_preds, axis=0)],
                figsize=(12, 6),
                legend=False,
                dump_path=os.path.join(exp.exp_dump_path, "media", "final.jpg"),
            )


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
