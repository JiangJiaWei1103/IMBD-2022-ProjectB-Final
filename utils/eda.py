"""
EDA utilities.
Author: JiaWei Jiang
"""
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import display
from matplotlib.axes import Axes

colors = sns.color_palette("Set2")


def summarize(
    df: pd.DataFrame,
    file_name: Optional[str] = None,
    n_rows_to_display: Optional[int] = 5,
) -> None:
    """Summarize DataFrame.

    Parameters:
        df: input data
        file_name: name of the input file
        n_rows_to_display: number of rows to display

    Return:
        None
    """
    file_name = "Data" if file_name is None else file_name

    # Derive NaN ratio for each column
    nan_ratio = pd.isna(df).sum() / len(df) * 100
    nan_ratio.sort_values(ascending=False, inplace=True)
    nan_ratio = nan_ratio.to_frame(name="NaN Ratio").T

    # Derive zero ratio for each column
    zero_ratio = (df == 0).sum() / len(df) * 100
    zero_ratio.sort_values(ascending=False, inplace=True)
    zero_ratio = zero_ratio.to_frame(name="Zero Ratio").T

    # Print out summarized information
    print(f"=====Summary of {file_name}=====")
    display(df.head(n_rows_to_display))
    print(f"Shape: {df.shape}")
    print("NaN ratio:")
    display(nan_ratio)
    print("Zero ratio:")
    display(zero_ratio)


def plot_pred_and_gt(
    y: np.ndarray,
    oof_preds: Union[List[np.ndarray], Dict[str, np.ndarray]],
    figsize: Tuple[int, int] = (6, 3),
    legend: bool = False,
    dump_path: str = "./result.jpg",
) -> None:
    """Plot prediction versus groundtruth.

    To facilitate model comparison, `oof_preds` can contain predictions
    from different models.
    """
    fig, ax = plt.subplots(figsize=figsize)
    xrange = np.arange(len(y))
    oof_preds_iter = oof_preds if isinstance(oof_preds, list) else oof_preds.items()

    # Wear cumulation
    for i, oof_pred in enumerate(oof_preds_iter):
        if isinstance(oof_pred, tuple):
            oof_pred_name, oof_pred_val = oof_pred[0], oof_pred[1]
        else:
            oof_pred_name, oof_pred_val = i, oof_pred
        ax.plot(xrange, oof_pred_val, "+-", label=f"Wear Cum Pred {oof_pred_name}")
    ax.plot(xrange, y, "bo-", label="Wear Cum Gt")
    ax.set_title("Wear Cumulation (Pred vs Gt)")
    ax.set_xlabel("Sample ID (Layer)")
    ax.set_xticks(xrange, xrange + 1)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Wear Cumulation")
    if legend:
        ax.legend()
    plt.savefig(dump_path)


def plot_univar_dist(
    data: Union[pd.Series, np.ndarray],
    feature: str,
    bins: int = 250,
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """Plot univariate distribution.

    Parameters:
        data: univariate data to plot
        feature: feature name of the data
        bins: number of bins
        ax: matplotlib Axes instance
        figsize: figure width and height in inches

    Return:
        None
    """
    show = False
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    if ax is None:
        assert figsize is not None, "`figsize` should be specified if ax isn't passed."
        fig, ax = plt.subplots(figsize=figsize)
        show = True
    sns.histplot(data=data, bins=bins, kde=True, palette=colors, ax=ax)
    ax.axvline(x=data.mean(), color="orange", linestyle="dotted", linewidth=1.5, label="Mean")
    ax.axvline(
        x=data.median(),
        color="green",
        linestyle="dotted",
        linewidth=1.5,
        label="Median",
    )
    ax.axvline(
        x=data.mode().values[0],
        color="red",
        linestyle="dotted",
        linewidth=1.5,
        label="Mode",
    )
    ax.set_title(
        f"{feature.upper()} Distibution\n"
        f"Min {data.min():.3f} | "
        f"Max {data.max():.3f} | "
        f"Skewness {data.skew():.3f} | "
        f"Kurtosis {data.kurtosis():.3f}"
    )
    ax.set_xlabel(f"{feature}")
    ax.set_ylabel("Bin Count")
    ax.legend()

    if show:
        plt.show()


def plot_series(
    df: pd.DataFrame,
    features: List[str],
    title: Optional[str] = None,
    x_axis: Optional[str] = None,
) -> None:
    """Plot series on the same figure.

    With multiple features plotted on the same figure, analyzers can
    observe if there's any synchronous behavior in each feature pair.

    Parameters:
        df: input data
        features: list of feature names
        title: title of the figure
        x_axis: name of the column acting as x axis

    Return:
        None
    """
    x = np.arange(len(df)) if x_axis is None else df[x_axis]

    fig = go.Figure()
    for f in features:
        fig.add_trace(go.Scatter(x=x, y=df[f], mode="lines", name=f))
    if title is not None:
        fig.update_layout(title=title)
    fig.show()


def plot_bivar(
    data: Union[pd.Series, np.ndarray],
    features: Optional[List[str]] = ["0", "1"],
) -> None:
    """Plot bivariate distribution with regression line fitted.

    Parameters:
        data: bivariate data to plot
        features: list of feature names

    Return:
        None
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    f1, f2 = features[0], features[1]
    corr = data[[f1, f2]].corr().iloc[0, 1]

    ax = sns.jointplot(
        x=data[f1],
        y=data[f2],
        kind="reg",
        height=6,
        marginal_ticks=True,
        joint_kws={"line_kws": {"color": "orange"}},
    )
    ax.fig.suptitle(f"{f1} versus {f2}, Corr={corr:.2}")
    ax.ax_joint.set_xlabel(f1)
    ax.ax_joint.set_ylabel(f2)
    plt.tight_layout()


def plot_nan_ratios(df: pd.DataFrame, xlabel: str = "Feature") -> None:
    """Plot NaN ratio bar plot ranked by ratio values.

    Parameters:
        df: input data
        xlabel: label on x-axis

    Return:
        None
    """
    nan_ratios = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    nan_ratios = nan_ratios[nan_ratios != 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=nan_ratios.index, y=nan_ratios.values, palette=colors, ax=ax)
    ax.set_title(f"NaN Ratios of Different {xlabel}s")
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=90, labelsize="small")
    ax.set_ylabel("NaN Ratio")
    plt.show()
