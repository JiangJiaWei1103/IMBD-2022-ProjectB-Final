"""Main script for training and evaluation processes using traditional ML."""
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
        dp = DataProcessor(dataset=args.dataset, data_type=args.data_type, infer=False, **exp.dp_cfg)
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

        if args.data_type == "normal":
            model_type = "normal"
        else:
            model_type = "aug" if args.mix_aug else "chunk"
        for sid, models_seed in enumerate(models):
            for mid, model in enumerate(models_seed):
                exp.dump_model(model, model_type=model_type, mid=f"seed{sid}_fold{mid}")

        # ===
        # Dump features
        with open(f"{exp.exp_dump_path}/features.txt", "w") as f:
            for feat in dp.get_feats():
                f.write(f"{feat}\n")
        # ===

        if LOCAL:
            plot_pred_and_gt(
                pd.read_csv("./dara/raw/train1/wear.csv")["MaxWear"].values,
                oof_preds,
                figsize=(12, 6),
                legend=False,
                exp_dump_path=exp.exp_dump_path,
            )


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
