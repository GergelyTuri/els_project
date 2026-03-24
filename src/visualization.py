"""
Visualization Functions for plotting Freezing data, Moseq data, and other data
"""

import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted

# Embed fonts in PDF/PS output (required for Adobe Illustrator compatibility)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Color palettes — divergent and colorblind-friendly
#   "default"  : Okabe-Ito blue / orange  (universal colorblind safe)
#   "palette2" : ColorBrewer PRGn purple / green
#   "palette3" : ColorBrewer RdBu red / blue
# ---------------------------------------------------------------------------
COLOR_PALETTES = {
    "default":  ["#0072B2", "#E69F00"],
    "palette2": ["#762A83", "#1B7837"],
    "palette3": ["#D6604D", "#4393C3"],
}

# ---------------------------------------------------------------------------
# Publication sizing: longer side = 5 inches
# The split-panel layout has a natural 5:3 (width:height) aspect ratio.
# ---------------------------------------------------------------------------
_LONG_SIDE = 5.0          # inches — the longer dimension of every saved figure
_ASPECT    = 5 / 3        # width : height for the two-panel freezing plot
_FIG_W     = _LONG_SIDE                 # 5.0 in
_FIG_H     = _LONG_SIDE / _ASPECT      # 3.0 in

# Font sizes scaled for a 5-inch publication figure
_FS_TICK   = 8
_FS_LABEL  = 9
_FS_TITLE  = 10
_FS_LEGEND = 8

# Line and marker weights
_LW        = 1.5   # plot line width (pt)
_MS        = 4     # marker size (pt)
_SPINE_LW  = 0.75  # axis spine / tick line width (pt)


def plot_freezing_time(
    data,
    variable,
    effect_size=None,
    pvalue=None,
    title_text="",
    hue="condition",
    ylim=[0, 60],
    output_filename="Freezing Time RM-ANOVA.svg",
    show_stats=False,
    palette="default",
):
    """
    Plots freezing time data for the SEFLA stage and subsequent stages
    side-by-side using a two-panel layout.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with at least 'day', the variable column, and the hue column.
    variable : str
        Column name of the freezing variable to plot (e.g. 'freezing_1s').
    effect_size : float, optional
        Effect size to annotate when show_stats=True.
    pvalue : float, optional
        p-value to annotate when show_stats=True.
    title_text : str
        Main figure title.
    hue : str
        Column used for color grouping (e.g. 'condition', 'Age', 'sex').
    ylim : list
        Y-axis limits. Default [0, 60].
    output_filename : str
        Output file path. Default 'Freezing Time RM-ANOVA.svg'.
    show_stats : bool
        Whether to annotate effect size and p-value.
    palette : str
        Color palette to use: 'default', 'palette2', or 'palette3'.
    """
    colors = COLOR_PALETTES.get(palette, COLOR_PALETTES["default"])

    SEFLA_data = data[data["day"] == "SEFLA"]
    subset_data = data[
        data["day"].isin(["SEFLB", "Recall 1", "Recall 2", "Recall 3", "Recall 4"])
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(_FIG_W, _FIG_H),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 4]},
    )

    # --- Left panel: SEFLA ---
    sns.pointplot(
        ax=ax1, data=SEFLA_data, x="day", y=variable, hue=hue,
        join=True, palette=colors, linewidth=_LW, markersize=_MS,
    )
    ax1.set_ylabel("Freezing Time (%)", fontsize=_FS_LABEL)
    ax1.set_ylim(ylim)
    ax1.set_xlabel("")
    ax1.tick_params(axis="both", labelsize=_FS_TICK, width=_SPINE_LW)
    ax1.legend().remove()
    sns.despine(ax=ax1)
    for spine in ax1.spines.values():
        spine.set_linewidth(_SPINE_LW)

    # --- Right panel: SEFLB through Recall 4 ---
    sns.pointplot(
        ax=ax2, data=subset_data, x="day", y=variable, hue=hue,
        join=True, palette=colors, linewidth=_LW, markersize=_MS,
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("Freezing Time (%)", fontsize=_FS_LABEL)
    ax2.tick_params(axis="both", labelsize=_FS_TICK, width=_SPINE_LW)
    legend = ax2.legend(fontsize=_FS_LEGEND, loc="upper right")
    legend.set_title(None)
    sns.despine(ax=ax2)
    for spine in ax2.spines.values():
        spine.set_linewidth(_SPINE_LW)

    fig.suptitle(title_text, fontsize=_FS_TITLE, x=0.5)

    if show_stats and effect_size is not None and pvalue is not None:
        pvalue_sign = "<" if pvalue < 0.05 else ">"
        ax2.text(
            0.5, 0.90, f"Effect size: {effect_size:.2f}",
            transform=ax2.transAxes, fontsize=_FS_TICK,
        )
        ax2.text(
            0.5, 0.82, f"p-value: {pvalue:.2e} {pvalue_sign} .05",
            transform=ax2.transAxes, fontsize=_FS_TICK,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, format="svg", dpi=300, bbox_inches="tight")
    plt.show()


def create_violin_plot(data, x, y, hue, title, ylabel, significant_syllables,
                       palette="default"):
    """
    Plots significant syllables as split violin plots.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        X-axis variable (e.g. syllable index).
    y : str
        Y-axis variable (e.g. angular velocity).
    hue : str
        Grouping variable.
    title : str
    ylabel : str
    significant_syllables : list
    palette : str
        Color palette: 'default', 'palette2', or 'palette3'.
    """
    colors = COLOR_PALETTES.get(palette, COLOR_PALETTES["default"])
    data = data[data["syllable"].isin(significant_syllables)]
    if data.empty:
        return print("No significant syllables to plot")

    fig_h = _LONG_SIDE
    fig_w = _LONG_SIDE * (4 / 3)   # landscape: longer side is width
    plt.figure(figsize=(fig_w, fig_h))
    sns.violinplot(data=data, x=x, y=y, hue=hue, split=True, inner="quartile",
                   palette=colors, linewidth=_LW)
    plt.title(title, fontsize=_FS_TITLE)
    plt.xlabel("Syllable Index", fontsize=_FS_LABEL)
    plt.ylabel(ylabel, fontsize=_FS_LABEL)
    plt.xticks(fontsize=_FS_TICK)
    plt.yticks(fontsize=_FS_TICK)
    plt.show()


def create_box_strip_plot(
    data,
    x,
    y,
    hue,
    title,
    ylabel,
    significant_syllables,
    syllable_map=None,
    ylim=None,
    output_filename=None,
    palette="default",
):
    """
    Plots significant syllables as box plots.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        X-axis variable (e.g. syllable index or name).
    y : str
        Y-axis variable.
    hue : str
        Grouping variable.
    title : str
    ylabel : str
    significant_syllables : list
    syllable_map : dict, optional
        Mapping from syllable index to name.
    ylim : list, optional
        Y-axis limits.
    output_filename : str, optional
        If provided, saves the figure to this path as SVG.
    palette : str
        Color palette: 'default', 'palette2', or 'palette3'.
    """
    colors = COLOR_PALETTES.get(palette, COLOR_PALETTES["default"])
    data = data[data["syllable"].isin(significant_syllables)]
    if data.empty:
        return print("No significant syllables to plot")

    if syllable_map and x == "syllable_name":
        x_order = natsorted([syllable_map[s] for s in significant_syllables])
    else:
        x_order = None

    fig_h = _LONG_SIDE
    fig_w = _LONG_SIDE * (4 / 3)
    plt.figure(figsize=(fig_w, fig_h))
    sns.boxplot(data=data, x=x, y=y, hue=hue, showfliers=False,
                order=x_order, palette=colors, linewidth=_LW)

    plt.title(title, fontsize=_FS_TITLE)
    plt.xlabel(None)
    plt.ylabel(ylabel, fontsize=_FS_LABEL)
    plt.xticks(fontsize=_FS_TICK, rotation=45)
    plt.yticks(fontsize=_FS_TICK)
    legend = plt.legend(fontsize=_FS_LEGEND, loc="upper right")
    legend.set_title(None)

    if ylim:
        plt.ylim(ylim)

    if output_filename:
        plt.savefig(output_filename, format="svg", dpi=300, bbox_inches="tight")

    plt.show()
