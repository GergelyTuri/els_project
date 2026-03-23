import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def find_freeze_transitions(freeze_frame_data):
    """
    Identifies the timestamps where the 'freeze' in the freezeframe transitions:
    - From 1 to 0 (offset, end of a freeze period)
    - From 0 to 1 (onset, start of a freeze period)

    Parameters:
    - freeze_frame_data: freeze frame DataFrame with a 'freeze' column

    Returns:
    - A DataFrame with columns ['t(sec)', 'transition_type', 'cohort_id', 'day'],
      sorted by 'cohort_id', 'day', and 't(sec)' (if available).
    """

    if "freeze" not in freeze_frame_data.columns:
        raise ValueError('Input DataFrame does not contain a "freeze" column')

    df = freeze_frame_data.copy()

    df["freeze_shifted"] = df["freeze"].shift(1, fill_value=0)

    offset_transitions = df[(df["freeze_shifted"] == 1) & (df["freeze"] == 0)][
        ["time"]
    ].copy()
    offset_transitions["transition_type"] = "offset"

    onset_transitions = df[(df["freeze_shifted"] == 0) & (df["freeze"] == 1)][
        ["time"]
    ].copy()
    onset_transitions["transition_type"] = "onset"

    for col in ["cohort_id", "day"]:
        if col in freeze_frame_data.columns:
            offset_transitions[col] = df.loc[offset_transitions.index, col].values
            onset_transitions[col] = df.loc[onset_transitions.index, col].values
        else:
            offset_transitions[col] = None
            onset_transitions[col] = None

    all_transitions = pd.concat(
        [offset_transitions, onset_transitions], ignore_index=True
    )

    sort_columns = [
        col for col in ["cohort_id", "day", "time"] if col in all_transitions.columns
    ]
    all_transitions = all_transitions.sort_values(by=sort_columns).reset_index(
        drop=True
    )

    return all_transitions


def calculate_median_freeze_duration(freeze_frame_data):
    """
    Calculates the overall median freeze duration across all mice and day

    Parameters:
    - freeze_frame_data: DataFrame, containing columns:
        - 't(sec)': time in seconds
        - 'freeze': binary column indicating whether the mouse is freezing
        - 'cohort)id': id for individual mouse
        - 'day': day number for the experiment

    Returns:
    - A float, the median freeze duration across all mice and day combinations
    """

    required_columns = ["t(sec)", "freeze", "cohort_id", "day"]
    if not required_columns.issubset(freeze_frame_data.columns):
        raise ValueError(
            f"Input Dataframe does not contain all required columns {required_columns}"
        )

    cohort_ids = freeze_frame_data["cohort_id"].unique()
    days = freeze_frame_data["day"].unique()

    all_freeze_durations = []

    for cohort_id in cohort_ids:
        for day in days:
            cohort_day_data = freeze_frame_data[
                (freeze_frame_data["cohort_id"] == cohort_id)
                & (freeze_frame_data["day"] == day)
            ]

            if cohort_day_data.empty:
                print(f"No data found for cohort_id: {cohort_id}, day: {day}")
                continue

            freeze_durations = []
            is_freezing = False

            for _, row in cohort_day_data.iterrows():
                if row["freeze"] == 1 and not is_freezing:
                    is_freezing = True
                    freeze_start = row["t(sec)"]
                elif row["freeze"] == 0 and is_freezing:
                    is_freezing = False
                    freeze_end = row["t(sec)"]
                    freeze_durations.append(freeze_end - freeze_start)

            if is_freezing and freeze_start is not None:
                freeze_end = cohort_day_data["t(sec)"].iloc[-1]
                freeze_durations.append(freeze_end - freeze_start)

            all_freeze_durations.extend(freeze_durations)

    overall_med = np.median(all_freeze_durations) if all_freeze_durations else 0
    return overall_med


def get_freeze_bouts(freeze_frame_data):
    """
    Extracts individual freezing bouts from the data, preserving additional columns
    (e.g., condition, sex, young, etc.) if they exist in the input DataFrame.

    Parameters:
        freeze_frame_data (pd.DataFrame): must contain columns:
            - 't(sec)'    : time in seconds
            - 'freeze'    : binary indicator (0 or 1)
            - 'cohort_id'
            - 'day'
          Optionally, it may contain other columns (e.g., 'condition', 'sex', 'young').

    Returns:
        A DataFrame where each row represents a freezing bout with:
            - 'cohort_id', 'day'
            - 'bout_start': time when freezing started
            - 'bout_end'  : time when freezing ended
            - 'duration'  : bout_end - bout_start
          plus any additional columns (like 'condition', 'sex', 'young')
          copied from the first row in each (cohort_id, day) group.
    """
    preserve_cols = ["condition", "sex", "young", "age"]  # Adjust as needed

    bouts = []

    grouped = freeze_frame_data.groupby(["cohort_id", "day"], as_index=False)

    for (cohort, day), group in grouped:
        group = group.sort_values("t(sec)")

        extra_data = {}
        for col in preserve_cols:
            if col in group.columns:
                extra_data[col] = group[col].iloc[0]

        trial_start = group["t(sec)"].iloc[0]
        trial_end = group["t(sec)"].iloc[-1]

        in_bout = False
        bout_start = None

        # Detect freezing bouts
        for _, row in group.iterrows():
            if row["freeze"] == 1 and not in_bout:
                # Start of a new bout
                in_bout = True
                bout_start = row["t(sec)"]
            elif row["freeze"] == 0 and in_bout:
                # End of the current bout
                bout_end = row["t(sec)"]
                bouts.append(
                    {
                        "cohort_id": cohort,
                        "day": day,
                        "bout_start": bout_start,
                        "bout_end": bout_end,
                        "duration": bout_end - bout_start,
                        **extra_data,  # Include preserved metadata
                    }
                )
                in_bout = False
                bout_start = None

        # If still in a bout at the end of the trial, close it
        if in_bout:
            bout_end = trial_end
            bouts.append(
                {
                    "cohort_id": cohort,
                    "day": day,
                    "bout_start": bout_start,
                    "bout_end": bout_end,
                    "duration": bout_end - bout_start,
                    **extra_data,
                }
            )

    return pd.DataFrame(bouts)


def compare_freeze_bout_lengths_by_minute(freeze_frame_data, total_experiment_time=300):
    """
    Bins freeze bouts by the minute in which they started and summarizes their duration.
    Preserves condition, sex, age, etc. from the updated get_freeze_bouts.

    Parameters:
        freeze_frame_data (pd.DataFrame): must contain columns:
            - 't(sec)': time in seconds
            - 'freeze': binary (0 or 1)
            - 'cohort_id': id for individual mouse
            - 'day': day number for the experiment
          plus any additional columns you want to preserve (e.g., 'condition', 'sex', 'young').
        total_experiment_time (int): total duration of the experiment in seconds (default 300)

    Returns:
        bouts_df (pd.DataFrame):
            Freeze bouts with columns:
                - 'cohort_id', 'day', 'bout_start', 'bout_end', 'duration', 'minute_bin'
                - and any preserved columns like 'condition', 'sex', 'young'.
        summary (dict):
            For each minute, a dict with:
                - 'bout_count'
                - 'median_duration'
                - 'mean_duration'
    """
    # Extract individual freeze bouts (now preserves additional columns)
    bouts_df = get_freeze_bouts(freeze_frame_data)

    # Bin freeze bouts by the minute in which they started
    bouts_df["minute_bin"] = (bouts_df["bout_start"] // 60 + 1).astype(int)

    # Only consider bouts that start within the experiment time
    max_minute = total_experiment_time // 60
    bouts_df = bouts_df[bouts_df["minute_bin"] <= max_minute]

    # Calculate summary statistics for each minute bin
    summary = {}
    for minute in range(1, max_minute + 1):
        minute_data = bouts_df[bouts_df["minute_bin"] == minute]
        summary[f"minute_{minute}"] = {
            "bout_count": len(minute_data),
            "median_duration": (
                minute_data["duration"].median() if not minute_data.empty else None
            ),
            "mean_duration": (
                minute_data["duration"].mean() if not minute_data.empty else None
            ),
        }

    return bouts_df, summary


def plot_freeze_duration_pointplot(bouts_df, group_cols, subset_col):
    """
    Creates multiple side-by-side pointplots of 'duration' by 'minute_bin',
    subsetting the data by a given column while using:
    - First group_col for color.
    - Second group_col for line style (if present).

    Parameters
    ----------
    bouts_df : pd.DataFrame
        The dataset containing:
        - 'minute_bin': numeric time bins
        - 'duration': freezing duration values
        - plus any columns to be used for grouping.
    group_cols : list of str (1 or 2 elements)
        First column determines color, second column (if present) determines line style.
    subset_col : str
        Column used to subset the data into multiple separate plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : list of matplotlib.axes.Axes
        The subplot axes.
    """
    if len(group_cols) < 1:
        raise ValueError("group_cols must have at least one column for color mapping.")

    color_key = group_cols[0]
    linestyle_key = group_cols[1] if len(group_cols) > 1 else None

    df = bouts_df.copy()
    df["minute_bin"] = df["minute_bin"].astype(int)

    unique_values = df[subset_col].dropna().unique()
    num_subplots = len(unique_values)

    if num_subplots == 0:
        raise ValueError(f"No valid values found in '{subset_col}'.")

    fig, axes = plt.subplots(
        ncols=num_subplots, figsize=(6 * num_subplots, 6), sharey=True
    )

    if num_subplots == 1:
        axes = [axes]

    color_palette = sns.color_palette("deep", n_colors=len(df[color_key].unique()))
    color_mapping = {
        key: color_palette[i] for i, key in enumerate(sorted(df[color_key].unique()))
    }

    linestyle_mapping = {}
    if linestyle_key:
        unique_linestyles = sorted(df[linestyle_key].unique())
        linestyle_mapping = {
            key: "-" if i == 0 else "--" for i, key in enumerate(unique_linestyles)
        }

    for ax, subset_value in zip(axes, unique_values):
        subset_df = df[df[subset_col] == subset_value]

        if linestyle_key:
            subset_df.loc[:, "interaction"] = (
                subset_df[color_key].astype(str)
                + "_"
                + subset_df[linestyle_key].astype(str)
            )
        else:
            subset_df.loc[:, "interaction"] = subset_df[color_key].astype(str)

        sorted_keys = sorted(subset_df["interaction"].unique())

        for interaction in sorted_keys:
            if linestyle_key:
                color_value, linestyle_value = interaction.split("_")
                linestyle = linestyle_mapping[linestyle_value]
            else:
                color_value = interaction
                linestyle = "-"

            data_subset = subset_df[subset_df["interaction"] == interaction]

            sns.pointplot(
                data=data_subset,
                x="minute_bin",
                y="duration",
                color=color_mapping[color_value],
                linestyles=linestyle,
                markers="o",
                ax=ax,
            )

        ax.set_title(f"Day: {subset_value}", fontsize=14)
        ax.set_xlabel("Minute Bin", fontsize=12)
        ax.set_ylabel("Freezing Duration (s)", fontsize=12)

        if ax == axes[-1]:  # Only show legend on the last (rightmost) subplot
            handles = []
            labels = []
            for key in sorted_keys:
                if linestyle_key:
                    color_val, linestyle_val = key.split("_")
                    linestyle = linestyle_mapping[linestyle_val]
                    label = f"{color_val} {linestyle_val}"
                else:
                    color_val = key
                    linestyle = "-"
                    label = color_val

                handle = plt.Line2D(
                    [0],
                    [0],
                    color=color_mapping[color_val],
                    linestyle=linestyle,
                    marker="o",
                )
                handles.append(handle)
                labels.append(label)

            ax.legend(
                handles,
                labels,
                title="Condition",
                loc="upper right",
                fontsize=10,
                title_fontsize=11,
            )
        else:
            ax.legend().set_visible(False)  # Hide legend on all other subplots

    sns.despine()
    plt.tight_layout()

    return fig, axes
