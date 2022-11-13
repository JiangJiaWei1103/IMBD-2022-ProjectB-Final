"""Experiment configuration logic."""
import os
import random
import string
from typing import Dict

import yaml

from paths import CONFIG_PATH


def gen_exp_id(model_name: str) -> str:
    """Generate unique experiment identifier.

    Parameters:
        model_name: name of model architecture

    Return:
        exp_id: experiment identifier
    """
    chars = string.ascii_lowercase + string.digits
    exp_id = "".join(random.SystemRandom().choice(chars) for _ in range(8))
    exp_id = f"{model_name}-{exp_id}"

    return exp_id


def setup_dp() -> Dict:
    """Return hyperparameters controlling data processing.

    Return:
        dp_cfg: hyperparameters of data processor
    """
    cfg_path = os.path.join(CONFIG_PATH, "dp.yaml")
    with open(cfg_path, "r") as f:
        dp_cfg = yaml.full_load(f)

    return dp_cfg


def setup_model(model_name: str) -> Dict:
    """Return hyperparameters of the specified model.

    Parameters:
        model_name: name of model architecture

    Return:
        model_cfg: hyperparameters of the specified model
    """
    cfg_path = os.path.join(CONFIG_PATH, f"model/{model_name}.yaml")
    with open(cfg_path, "r") as f:
        model_cfg = yaml.full_load(f)

    return model_cfg
