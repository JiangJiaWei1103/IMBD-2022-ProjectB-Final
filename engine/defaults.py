"""
Boilerplate logic for controlling training and evaluation processes.
Author: JiaWei Jiang
"""
import argparse
from abc import abstractmethod
from argparse import Namespace
from typing import Optional

__all__ = ["TrainEvalArgParser", "InferArgParser"]


class BaseArgParser:
    """Base class for all customized argument parsers.

    =Todo=
    Move to base directory.
    """

    def __init__(self) -> None:
        self.argparser = argparse.ArgumentParser()
        self._build()

    def parse(self) -> Namespace:
        """Return arguments driving the designated processes.

        Return:
            args: arguments driving the designated processes
        """
        args = self.argparser.parse_args()
        return args

    @abstractmethod
    def _build(self) -> None:
        """Build argument parser."""
        raise NotImplementedError

    def _str2bool(self, arg: str) -> Optional[bool]:
        """Convert boolean argument from string representation into
        bool.

        Parameters:
            arg: argument in string representation

        Return:
            True or False: argument in bool dtype
        """
        # See https://stackoverflow.com/questions/15008758/
        # parsing-boolean-values-with-argparse
        if isinstance(arg, bool):
            return arg
        if arg.lower() in ("true", "t", "yes", "y", "1"):
            return True
        elif arg.lower() in ("false", "f", "no", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Expect boolean representation " "for argument --eval-only.")


class TrainEvalArgParser(BaseArgParser):
    """Argument parser for training and evaluation processes."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--exp-id",
            type=str,
            default=None,
            help="experiment identifier",
        )
        self.argparser.add_argument(
            "--dataset",
            type=str,
            choices=["train1", "train2", "mix"],
            default=None,
            help="name of the dataset",
        )
        self.argparser.add_argument(
            "--data-type",
            type=str,
            choices=["normal", "aug"],
            default=None,
            help="type of the data",
        )
        self.argparser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="name of the model architecture",
        )
        self.argparser.add_argument(
            "--mix-aug",
            type=self._str2bool,
            default=None,
            help="if True, models are trained on data chunks mixed together",
        )
        self.argparser.add_argument("--n-seeds", type=int, default=None, help="total number of seeds")
        self.argparser.add_argument("--n-folds", type=int, default=None, help="total number of folds")


class InferArgParser(BaseArgParser):
    """Argument parser for inference process."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--exp-id",
            type=str,
            default=None,
            help="experiment identifier of pre-trained models to use",
        )
        self.argparser.add_argument(
            "--dataset",
            type=str,
            choices=["train1", "train2", "mix", "test"],
            default=None,
            help="name of the dataset",
        )
        self.argparser.add_argument(
            "--eval-holdout",
            type=self._str2bool,
            default=None,
            help="run evaluation on holdout or not",
        )
