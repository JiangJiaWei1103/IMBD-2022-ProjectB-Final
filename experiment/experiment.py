"""
Experiment logger.

This file contains the definition of experiment logger for experiment
configuration, message logging, object dumping, etc.
"""
from __future__ import annotations

import logging
import os
import pickle
from argparse import Namespace
from types import TracebackType
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator

from config.config import gen_exp_id, setup_dp, setup_model
from paths import DUMP_PATH
from utils.logger import Logger


class Experiment(object):
    """Experiment logger.

    Parameters:
        args: arguments driving training and evaluation processes
    """

    cfg: Dict[str, Dict[str, Any]]
    model_params: Dict[str, Any]
    fit_params: Dict[str, Any]
    exp_dump_path: str

    def __init__(self, args: Namespace):
        if args.exp_id is None:
            self.exp_id = gen_exp_id(args.model_name)
        else:
            self.exp_id = args.exp_id
        self.args = args
        self.dp_cfg = setup_dp()
        self.model_cfg = setup_model(args.model_name)

        self._parse_model_cfg()
        self._agg_cfg()
        self._mkbuf()

        # Setup experiment logger
        self.logger = Logger(logging_file=os.path.join(self.exp_dump_path, "exp.log")).get_logger()

    def __enter__(self) -> Experiment:
        self._run()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_inst: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._halt()

    def dump_cfg(self, cfg: Dict[str, Any], file_name: str) -> None:
        """Dump config dictionary to corresponding path.

        Parameters:
            cfg: configuration
            file_name: name of the file with .yaml extension

        Return:
            None
        """
        dump_path = os.path.join(self.exp_dump_path, "config", f"{file_name}.yaml")
        with open(dump_path, "w") as f:
            yaml.dump(cfg, f)

    def dump_ndarr(self, arr: np.ndarray, file_name: str) -> None:
        """Dump np.ndarray under corresponding path.

        Parameters:
            arr: array to dump
            file_name: name of the file with .npy extension

        Return:
            None
        """
        dump_path = os.path.join(self.exp_dump_path, "preds", file_name)
        np.save(dump_path, arr)

    def dump_df(self, df: pd.DataFrame, file_name: str, ext: str = "csv") -> None:
        """Dump DataFrame under corresponding path.

        Support only for dumping feature importance df now.

        Parameters:
            file_name: name of the file with . extension
        """
        dump_path = os.path.join(self.exp_dump_path, f"{file_name}.{ext}")
        df.to_csv(dump_path, index=False)

    def dump_model(self, model: BaseEstimator, model_type: str, mid: str) -> None:
        """Dump estimator to corresponding path.

        Parameters:
            model: well-trained estimator
            model_type: type of the model, the choices are as follows:
                {"normal", "aug", "chunk"}
            mid: identifer of the model

        Return:
            None
        """
        dump_dir = os.path.join(self.exp_dump_path, "models", model_type)
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        dump_path = os.path.join(dump_dir, f"{mid}.pkl")
        with open(dump_path, "wb") as f:
            pickle.dump(model, f)

    def _parse_model_cfg(self) -> None:
        """Configure model parameters and parameters passed to `fit`
        method if they're provided.
        """
        self.model_params = self.model_cfg["model_params"]
        if self.model_cfg["fit_params"] is not None:
            self.fit_params = self.model_cfg["fit_params"]

    def _agg_cfg(self) -> None:
        """Aggregate sub configurations of different components into
        one summarized configuration.
        """
        self.cfg = {
            "common": vars(self.args),
            "dp": self.dp_cfg,
            "model": self.model_params,
            "fit": self.fit_params,
        }

    def _mkbuf(self) -> None:
        """Make local buffer for experiment output dumping."""
        if not os.path.exists(DUMP_PATH):
            os.mkdir(DUMP_PATH)
        self.exp_dump_path = os.path.join(DUMP_PATH, self.exp_id)
        os.mkdir(self.exp_dump_path)

        # Create folders for output objects
        os.mkdir(os.path.join(self.exp_dump_path, "config"))
        os.mkdir(os.path.join(self.exp_dump_path, "models"))
        os.mkdir(os.path.join(self.exp_dump_path, "preds"))
        os.mkdir(os.path.join(self.exp_dump_path, "imp"))

    def _run(self) -> None:
        """Start a new experiment entry."""
        self._log_exp_metadata()

    def _log_exp_metadata(self) -> None:
        """Log metadata of the experiment to Wandb."""
        logging.info(f"=====Experiment {self.exp_id}=====")
        logging.info(f"-> CFG: {self.cfg}\n")

    def _halt(self) -> None:
        logging.info(f"=====End of Experiment {self.exp_id}=====")
