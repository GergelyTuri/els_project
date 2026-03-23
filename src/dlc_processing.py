import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import pingouin as pg
from pingouin import power_anova


def dlc_to_long(file_path):
    """
    Transforms wide DLC data into a long format, using metadata rows to structure columns
    and extracts cohort_id from the file name.

    Parameters:
    - file_path: String representing the file path to the wide-format positional data CSV.

    Returns:
    - long_data: DataFrame in long format with columns ['x', 'y', 'likelihood', 'body_part', 'coords', 'cohort_id'].
    """

    # Extract the file name from the file path
    file_name = os.path.basename(file_path)

    # Attempt to match both formats
    match1 = re.match(r"(\w+)_([a-zA-Z]+\d*)_(\d+)", file_name)  # Format 1
    match2 = re.match(r"(\w+)_([a-zA-Z]+\d*)_(\d+)-(\d+)", file_name)  # Format 2

    if (
        match2
    ):  # only for ptsd9 group (ptsd9 group have longer cohort_id such as ptsd9_31_2)
        cohort_prefix = match2.group(1)  # 'ptsd9'
        day = match2.group(2)  # 'recall4'
        cohort_number1 = match2.group(3)  # '31'
        cohort_number2 = match2.group(4)  # '2'
        cohort_id = f"{cohort_prefix}_{cohort_number1}_{cohort_number2}"  # 'ptsd9_31_2'
    elif match1:  # for all other groups
        cohort_prefix = match1.group(1)  # 'ptsd2'
        day = match1.group(2)  # 'recall1'
        cohort_number = match1.group(3)  # '81'
        cohort_id = f"{cohort_prefix}_{cohort_number}"  # 'ptsd2_81'
    else:
        cohort_id = "unknown"
        day = "unknown"

    raw_data = pd.read_csv(file_path)

    # Extract body parts from the first metadata row and coordinate types from the second row
    body_parts = raw_data.iloc[0, 1::3].values
    coordinates = ["x", "y", "likelihood"]

    # Generate column names using body parts and coordinates
    column_names = [f"{body}_{coord}" for body in body_parts for coord in coordinates]

    # Reload the dataset, skipping metadata rows and assigning correct column names
    data = pd.read_csv(file_path, skiprows=[0, 1])
    data = data.iloc[:, 1:]  # Drop the first unnecessary 'coords' column
    data.columns = column_names

    # Insert 'cohort_id' and 'day' column
    data["cohort_id"] = cohort_id
    data["day"] = day

    # Convert to long format
    long_data = pd.DataFrame()
    for part in body_parts:
        part_data = data[
            [f"{part}_x", f"{part}_y", f"{part}_likelihood", "cohort_id", "day"]
        ].copy()
        part_data.columns = ["x", "y", "likelihood", "cohort_id", "day"]
        part_data["body_part"] = part
        part_data["index"] = part_data.index
        part_data["t(sec)"] = (part_data["index"] / len(part_data) * 300).round(2)
        long_data = pd.concat([long_data, part_data], ignore_index=True)

    return long_data


def process_dlc_folder(folder_path):
    """
    Processes all relevant DLC CSV files in a folder, converts them to long format, and
    concatenates them into a single DataFrame.

    Parameters:
    - folder_path: String representing the path to the folder containing DLC CSV files.

    Returns:
    - combined_data: DataFrame containing all processed data in long format.
    """
    combined_data = pd.DataFrame()

    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter for relevant DLC CSV files
    dlc_files = [f for f in all_files if f.endswith(".csv") and "DLC" in f]

    successful_count = 0

    # Process each file
    for file_name in dlc_files:
        file_path = os.path.join(folder_path, file_name)
        # Process the file using dlc_to_long
        try:
            long_data = dlc_to_long(file_path)
            combined_data = pd.concat([combined_data, long_data], ignore_index=True)
            successful_count += 1
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    print(f"Successfully processed {successful_count} files.")

    return combined_data


def check_dlc_shapes(folder_path):
    """
    Checks all relevant DLC CSV files in a folder for the expected shape of (1124, 8).
    If a file does not match this shape, prints the file name and its shape.

    Parameters:
    - folder_path: String representing the path to the folder containing DLC CSV files.
    """
    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter for relevant DLC CSV files
    dlc_files = [f for f in all_files if f.endswith(".csv") and "DLC" in f]

    successful_count = 0
    outlier_files = []

    # Process each file
    for file_name in dlc_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            long_data = dlc_to_long(file_path)
            if long_data.shape == (13488, 8):
                successful_count += 1
            else:
                outlier_files.append((file_name, long_data.shape))
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    print(
        f"Successfully processed {successful_count} files with the expected shape (1124, 8)."
    )
    if outlier_files:
        print("Files with unexpected shapes:")
        for file_name, shape in outlier_files:
            print(f"  - {file_name}: {shape}")
    else:
        print("All files have the expected shape.")


def compute_body_length(df, body_parts):
    """
    Computes body length by summing Euclidean distances between consecutive keypoints along the spine.

    Parameters:
    - df: DataFrame containing columns ['x', 'y', 'body_part', 'cohort_id', 'day', 't(sec)']
    - body_parts: List of keypoints defining the body axis, ordered from head to tail.

    Returns:
    - length_df: DataFrame containing body length per frame.
    """
    length_list = []

    for t in df["t(sec)"].unique():
        frame_data = df[df["t(sec)"] == t].set_index("body_part")

        length = 0
        for i in range(len(body_parts) - 1):
            p1, p2 = body_parts[i], body_parts[i + 1]

            if p1 in frame_data.index and p2 in frame_data.index:
                coords1 = frame_data.loc[p1, ["x", "y"]].values
                coords2 = frame_data.loc[p2, ["x", "y"]].values

                if coords1.shape[0] == 2:  # Ensure only two values (x, y) are retrieved
                    x1, y1 = coords1
                    x2, y2 = coords2
                    length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                else:
                    print(
                        f"Warning: Unexpected shape {coords1.shape} for {p1} at time {t}"
                    )

        length_list.append({"t(sec)": t, "body_length": length})

    length_df = pd.DataFrame(length_list)
    return length_df


def compute_spine_curvature(df, body_parts):
    """
    Computes spine curvature by summing angles between consecutive spine segments.

    Parameters:
    - df: DataFrame with columns ['x', 'y', 'body_part', 'cohort_id', 'day', 't(sec)']
    - body_parts: Ordered list of keypoints defining the spine.

    Returns:
    - curvature_df: DataFrame containing spine curvature per frame.
    """
    curvature_list = []

    for t in df["t(sec)"].unique():
        frame_data = df[df["t(sec)"] == t].set_index("body_part")

        angles = []
        missing_parts = []

        for i in range(len(body_parts) - 2):
            p1, p2, p3 = body_parts[i], body_parts[i + 1], body_parts[i + 2]

            if (
                p1 in frame_data.index
                and p2 in frame_data.index
                and p3 in frame_data.index
            ):
                x1, y1 = frame_data.loc[p1, ["x", "y"]].values
                x2, y2 = frame_data.loc[p2, ["x", "y"]].values
                x3, y3 = frame_data.loc[p3, ["x", "y"]].values

                v1 = np.array([x2 - x1, y2 - y1])
                v2 = np.array([x3 - x2, y3 - y2])

                dot_product = np.dot(v1, v2)
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

                if norm_product > 0:
                    theta = np.arccos(dot_product / norm_product)
                    angles.append(theta)
                else:
                    angles.append(0)
            else:
                missing_parts.append((p1, p2, p3))

        if missing_parts:
            print(f"Missing keypoints at time {t}: {missing_parts}")

        # Sum all angles for total curvature
        total_curvature = np.sum(angles)

        curvature_list.append({"t(sec)": t, "spine_curvature": total_curvature})

    curvature_df = pd.DataFrame(curvature_list)
    return curvature_df


def plot_kinematics_pointplot(bouts_df, group_cols, x_col="day", y_col="body_length"):
    """
    Creates a point plot of `y_col` by `x_col`,
    using:
    - First group_col for color.
    - Second group_col for line style (if present).

    Parameters
    ----------
    bouts_df : pd.DataFrame
        The dataset containing:
        - `x_col`: categorical or numeric x-axis variable
        - `y_col`: dependent variable to be plotted
        - plus any columns to be used for grouping.
    group_cols : list of str (1 or 2 elements)
        First column determines color, second column (if present) determines line style.
    x_col : str, optional
        The column to use for the x-axis (default: "day").
    y_col : str, optional
        The column to use for the y-axis (default: "body_length").

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The plot axis.
    """
    if len(group_cols) < 1:
        raise ValueError("group_cols must have at least one column for color mapping.")

    color_key = group_cols[0]
    linestyle_key = group_cols[1] if len(group_cols) > 1 else None

    df = bouts_df.copy()

    # Drop None values from color_key and linestyle_key before sorting
    df = df.dropna(subset=[color_key] + ([linestyle_key] if linestyle_key else []))

    # Ensure values are sorted without NoneType issues
    unique_colors = sorted([val for val in df[color_key].unique() if pd.notna(val)])

    color_palette = sns.color_palette("deep", n_colors=len(unique_colors))
    color_mapping = {key: color_palette[i] for i, key in enumerate(unique_colors)}

    linestyle_mapping = {}
    if linestyle_key:
        unique_linestyles = sorted(
            [val for val in df[linestyle_key].unique() if pd.notna(val)]
        )
        linestyle_mapping = {
            key: "-" if i == 0 else "--" for i, key in enumerate(unique_linestyles)
        }

    if linestyle_key:
        df["interaction"] = (
            df[color_key].astype(str) + "x" + df[linestyle_key].astype(str)
        )
    else:
        df["interaction"] = df[color_key].astype(str)

    sorted_keys = sorted(df["interaction"].unique())

    fig, ax = plt.subplots(figsize=(8, 6))

    for interaction in sorted_keys:
        if linestyle_key:
            color_value, linestyle_value = interaction.split("x")
            linestyle = linestyle_mapping.get(linestyle_value, "-")
        else:
            color_value = interaction
            linestyle = "-"

        data_subset = df[df["interaction"] == interaction]

        sns.pointplot(
            data=data_subset,
            x=x_col,
            y=y_col,
            color=color_mapping.get(color_value, "gray"),
            linestyles=linestyle,
            markers="o",
            ax=ax,
        )

    ax.set_xlabel(x_col.capitalize(), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").capitalize(), fontsize=12)

    # Capitalized legend labels
    handles = []
    labels = []
    for key in sorted_keys:
        if linestyle_key:
            color_val, linestyle_val = key.split("x")
            linestyle = linestyle_mapping.get(linestyle_val, "-")
            label = f"{color_val.capitalize()} {linestyle_val.capitalize()}"
        else:
            color_val = key
            linestyle = "-"
            label = color_val.capitalize()

        handle = plt.Line2D(
            [0],
            [0],
            color=color_mapping.get(color_val, "gray"),
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

    sns.despine()
    plt.tight_layout()

    return fig, ax


class kinematicsAnalysis:
    def __init__(self, data):
        """
        Initialize the class with necessary data.
        """
        self.data = data

    def repeated_measures_anova(
        self, dv, within, between, subject, power=0.80, groups=2
    ):
        """
        Perform repeated measures ANOVA and calculate required sample size.

        Parameters:
            dv (str): Dependent variable column name.
            within (str): Within-subject factor column name.
            between (str): Between-subject factor column name.
            subject (str): Subject identifier column name.
            power (float): Desired statistical power (default: 0.80).
            groups (int): Number of groups (default: 2).

        Returns:
            dict: Results including ANOVA summary and required sample size per group.
        """
        # Perform the repeated measures ANOVA
        anova_result = pg.mixed_anova(
            data=self.data,
            dv=dv,
            within=within,
            between=between,
            subject=subject,
            effsize="np2",
        )

        # Extract eta squared from the ANOVA result
        eta_squared = anova_result["np2"].iloc[0]

        # Calculate the required sample size per group
        required_sample_size = power_anova(
            eta_squared=eta_squared, k=groups, power=power
        )

        return {
            "anova_result": anova_result,
            "required_sample_size": required_sample_size,
        }

    def post_hoc_analysis_significant(
        self, dv, within, between, subject, p_adjust="bonferroni", alpha=0.05
    ):
        """
        Perform post hoc pairwise comparisons and display only significant results.

        Parameters:
            dv (str): Dependent variable column name.
            within (str): Within-subject factor column name (e.g., day).
            between (str): Between-subject factor column name (e.g., group).
            subject (str): Subject identifier column name.
            p_adjust (str): Method for p-value adjustment (default: 'bonferroni').
            alpha (float): Significance level for filtering results (default: 0.05).

        Returns:
            DataFrame: Filtered table showing only significant comparisons.
        """

        # Perform pairwise t-tests
        pairwise_results = pg.pairwise_tests(
            data=self.data,
            dv=dv,
            between=between,
            within=within,
            subject=subject,
            padjust=p_adjust,
            effsize="cohen",
        )

        # Filter for significant results
        significant_results = pairwise_results[pairwise_results["p-corr"] < alpha]

        # Select relevant columns
        significant_results = significant_results[
            ["Contrast", "day", "A", "B", "p-corr", "cohen"]
        ]

        # Rename columns
        significant_results.rename(columns={"cohen": "Cohen'd"}, inplace=True)

        return significant_results
