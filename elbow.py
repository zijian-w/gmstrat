from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from hierachical import HClusters


def compute_elbow(sp, k_values: Iterable[int]) -> pd.DataFrame:
    hc = HClusters(sp)
    districts = sp.get_all_districts()
    rows = []

    for k in tqdm(list(k_values), desc="elbow"):
        hc.update_clusters(k)
        wcss = hc._compute_wcss_centroid(districts, hc.i2c)
        rows.append({"k": k, "wcss": wcss})

    df = pd.DataFrame(rows)
    df["wcss_normalized"] = df["wcss"] / df["wcss"].iloc[0]
    df["improvement"] = -df["wcss_normalized"].diff()
    return df


def plot_elbow(df: pd.DataFrame, selected_k: int = 34):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axes[0].plot(df["k"], df["wcss_normalized"])
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("normalized dispersion")

    axes[1].plot(df["k"], df["improvement"], marker="o", markersize=3)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("marginal improvement")

    for ax in axes:
        ax.axvline(selected_k, color="gray", linestyle="--", label=f"K = {selected_k}")
        ax.legend()

    fig.tight_layout()
    return fig, axes
