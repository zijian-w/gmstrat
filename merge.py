from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter


def plot_merges(sp, hc, gdf, k_values=(33, 34, 35)):
    all_districts = sp.get_all_districts()
    rows = []
    all_maps = []

    for k in k_values:
        left, right, merged = hc.merge_from_k(k)
        groups = [left, right, merged]
        maps = []
        wcss = []

        for group in groups:
            counts = np.zeros(sp.num_precincts, dtype=float)
            for district_uid in group:
                counts[all_districts[district_uid]] += 1
            density = 100 * counts / len(group)
            centroid = np.flatnonzero(density >= 50)
            maps.append(density)
            wcss.append(
                sum(
                    sp.compute_distance(all_districts[district_uid], centroid)
                    for district_uid in group
                )
            )

        rows.append(
            {
                "k": k,
                "left_size": len(left),
                "right_size": len(right),
                "improvement": np.log10(wcss[2] - wcss[0] - wcss[1]),
                "maps": maps,
            }
        )
        all_maps.extend(maps)

    norm = Normalize(
        vmin=min(values.min() for values in all_maps),
        vmax=max(values.max() for values in all_maps),
    )
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(18, 5 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, row in enumerate(rows):
        for col_idx, (values, title) in enumerate(
            zip(row["maps"], ["Left", "Right", "Merged"])
        ):
            ax = axes[row_idx, col_idx]
            gdf.assign(_value=values).plot(
                column="_value",
                cmap="Blues",
                norm=norm,
                ax=ax,
                edgecolor="black",
                linewidth=0.2,
            )
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.text(
                    -0.06,
                    0.5,
                    f"K={row['k']}\n|L|={row['left_size']}  |R|={row['right_size']}\n"
                    rf"$\log \Delta={row['improvement']:.3g}$",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                )

    colorbar = fig.colorbar(
        ScalarMappable(cmap="Blues", norm=norm),
        ax=axes.ravel().tolist(),
        fraction=0.03,
        pad=0.02,
    )
    colorbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, position: f"{value:.2f}%")
    )
    return fig, axes
