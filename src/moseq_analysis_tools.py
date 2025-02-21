import h5py
import pandas as pd
from freeze_analysis_tools import find_freeze_transitions
import re 
from sklearn.metrics import f1_score, recall_score

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score


class MoseqProcessor:

    def __init__(self, file_path):
        """
        Initializes the Keypoint-Moseq processor class 

        Parameters:
        file_path (str): The path to the .h5 file containing the moseq data
        """

        self.file_path = file_path
        self.data = None 

    def _extract_cohort_and_day(self, group_name):
        """
        Extracts the cohort ID and day from the group name.

        Parameters:
            group_name (str): The group name from the h5 file.
        
        Returns:
            tuple: (cohort_id, day)
        """
        match1 = re.match(r'(\w+)_([a-zA-Z]+\d*)_(\d+)', group_name)  # Format 1
        match2 = re.match(r'(\w+)_([a-zA-Z]+\d*)_(\d+)-(\d+)', group_name)  # Format 2

        if match2:  # Format 2 (e.g., ptsd9 group with longer cohort_id)
            cohort_prefix = match2.group(1)
            day = match2.group(2)
            cohort_number1 = match2.group(3)  # First cohort number
            cohort_number2 = match2.group(4)  # Second cohort number
            cohort_id = f"{cohort_prefix}_{cohort_number1}_{cohort_number2}"  # Maintain format as "ptsd9_28"
        elif match1:  # Format 1 (standard format)
            cohort_prefix = match1.group(1)
            day = match1.group(2)
            cohort_number = match1.group(3)
            cohort_id = f"{cohort_prefix}_{cohort_number}"
        else:
            cohort_id = 'unknown'
            day = 'unknown'

        return cohort_id, day

    def load_data(self):
        """
        Loads MoSeq data from the h5 file and stores it in the 'data' attribute.
        
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        
        all_data = []

        with h5py.File(self.file_path, 'r') as hdf:
            for group_name in hdf.keys():
                group_data = hdf[group_name]

                # Extract data arrays
                centroid = group_data['centroid'][:]
                heading = group_data['heading'][:]
                latent_state = group_data['latent_state'][:]
                syllable = group_data['syllable'][:]

                # Create the main DataFrame
                df = pd.DataFrame({
                    'centroid_x': centroid[:, 0],
                    'centroid_y': centroid[:, 1],
                    'heading': heading,
                    'syllable': syllable
                })

                # Create and concatenate the latent state DataFrame
                latent_df = pd.DataFrame(latent_state, 
                                         columns=[f'latent_{i}' for i in 
                                                  range(latent_state.shape[1])])
                df = pd.concat([df, latent_df], axis=1)

                # Extract cohort_id and day
                cohort_id, day = self._extract_cohort_and_day(group_name)
                df['cohort_id'] = cohort_id
                df['day'] = day


                all_data.append(df)

        self.data = pd.concat(all_data, ignore_index=True)
        return self.data
    
def create_time_column(df, frame_rate):
    """
    Adds a time column to the DataFrame based on frame number and frame rate,
    ensuring each (cohort_id, day) group is reindexed separately.

    Parameters:
        df (pd.DataFrame): The DataFrame containing cohort_id and day.
        frame_rate (float): The frame rate to calculate time per frame.

    Returns:
        pd.DataFrame: The DataFrame with an added 'time' column.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None. Provide a valid dataset.")

    df = df.copy()  # Avoid modifying the original DataFrame

    # Loop through each unique (cohort_id, day) combination
    for (cohort, day), group in df.groupby(['cohort_id', 'day']):
        if group.empty:
            continue  # Skip empty groups
        
        # Reset index within the group to create a unique frame count per trial
        reindexed_group = group.reset_index(drop=True)
        
        # Assign time as the new index divided by the frame rate
        df.loc[group.index, 'time'] = reindexed_group.index / frame_rate

    return df

def extract_moseq_windows(kp_data, freeze_transitions, window_size=3):
    """
    Extracts MoSeq data surrounding freeze transitions.

    Parameters:
    kp_data (pd.DataFrame): Processed MoSeq data with 'cohort_id', 'day', and 'time' columns.
    freeze_transitions (pd.DataFrame): DataFrame containing freeze transition data.
            Must have columns ['time', 'cohort_id', 'day', 'transition_type'].
    window_size (int, optional): Number of seconds before and after the transition time to extract (default: 3).

    Returns:
        pd.DataFrame: A DataFrame containing the extracted MoSeq data with the specified columns.
    """
    required_cols = ['time', 'cohort_id', 'day', 'transition_type']
    if not all(col in freeze_transitions.columns for col in required_cols):
        raise ValueError(f"freeze_transitions must contain columns: {required_cols}")

    selected_columns = ['syllable', 'cohort_id', 'day', 'time', 
                        'relative_time', 'transition_type']
    result_data = []

    # Iterate through each transition in freeze_transitions
    for _, transition in freeze_transitions.iterrows():
        cohort_id = transition['cohort_id']
        day = transition['day']
        transition_time = transition['time']
        transition_type = transition['transition_type']

        # Subset kp_data by cohort_id and day
        subset = kp_data[(kp_data['cohort_id'] == cohort_id) & (kp_data['day'] == day)]

        if subset.empty:
            continue  # Skip if no matching data

        # Find the closest matching time point
        closest_index = (subset['time'] - transition_time).abs().idxmin()
        closest_time = subset.loc[closest_index, 'time']

        # Extract window of ±window_size seconds
        window_data = subset[(subset['time'] >= closest_time - window_size) & 
                             (subset['time'] <= closest_time + window_size)].copy()

        # Assign transition type to the window
        window_data['transition_type'] = transition_type

        # Compute relative time
        window_data['relative_time'] = window_data['time'] - closest_time

        # Keep only relevant columns
        result_data.append(window_data[selected_columns])

    # Combine all extracted data into a single DataFrame
    final_df = pd.concat(result_data, ignore_index=True) if result_data else pd.DataFrame(columns=selected_columns)

    return final_df


def plot_freeze_ethogram(freeze_df, moseq_df, freeze_col='freeze', moseq_col='moseq_freeze', 
                         time_col='time', freeze_label='FreezeFrame', moseq_label='KPMS'):
    """
    Plots an ethogram comparing freeze and moving states between two models using actual time values.
    Also computes F1-score and Sensitivity (recall).

    Parameters:
    - freeze_df: DataFrame containing the FreezeFrame data, must include a 'time' column.
    - moseq_df: DataFrame containing the KPMS data, must include a 'time' column.
    - freeze_col: Column name in freeze_df representing the freeze states (default is 'freeze').
    - moseq_col: Column name in moseq_df representing the freeze states (default is 'moseq_freeze').
    - time_col: Column name representing the time axis in both DataFrames.
    - freeze_label: Label for the FreezeFrame data (default is 'FreezeFrame').
    - moseq_label: Label for the KPMS data (default is 'KPMS').
    """

    freeze_len = len(freeze_df)
    moseq_len = len(moseq_df)

    # Align lengths of freeze and moseq data
    min_length = min(freeze_len, moseq_len)
    freeze_states = freeze_df[freeze_col].iloc[:min_length]
    moseq_states = moseq_df[moseq_col].iloc[:min_length]

    # Use actual time values
    freeze_time = freeze_df[time_col].iloc[:min_length]
    moseq_time = moseq_df[time_col].iloc[:min_length]

    # Calculate F1 score & Sensitivity (Recall)
    try:
        f1 = f1_score(freeze_states, moseq_states, average='binary')
        sensitivity = recall_score(freeze_states, moseq_states)  # TP / (TP + FN)
    except ValueError as e:
        print(f"Error calculating F1 score or Sensitivity: {e}")
        f1, sensitivity = 0.0, 0.0

    fig, ax = plt.subplots(figsize=(10, 4))

    moving_color = "#F08080"  
    freeze_color = "#4682B4"  

    # Plot FreezeFrame data
    ax.fill_between(freeze_time, 0, 1, color=moving_color, alpha=0.5)
    ax.fill_between(freeze_time, 0, 1, where=(freeze_states == 1), 
                    step='post', color=freeze_color, alpha=0.8)

    # Plot KPMS data
    ax.fill_between(moseq_time, 1, 2, color=moving_color, alpha=0.5)
    ax.fill_between(moseq_time, 1, 2, where=(moseq_states == 1), 
                    step='post', color=freeze_color, alpha=0.8)

    # Add black outlines for each row
    for y in [0, 1, 1, 2]:
        ax.plot([freeze_time.iloc[0], freeze_time.iloc[-1]], [y, y], color='black', linewidth=1.5)

    # Extract metadata for title
    cohort_id = freeze_df['cohort_id'].iloc[0]
    day = freeze_df['day'].iloc[0]

    # Set labels and titles
    ax.set_yticks([0.45, 1.55])
    ax.set_yticklabels([freeze_label, moseq_label])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Ethogram {cohort_id} {day}: {freeze_label} vs {moseq_label}\nF1 Score: {f1:.2f} | Sensitivity: {sensitivity:.2f}")

    plt.tight_layout()
    plt.show()


def plot_all_ethograms(freeze_frame, moseq_data, plot_func=plot_freeze_ethogram, time_col='time'):
    """
    Matches unique cohort IDs between freeze_frame and moseq_data datasets,
    aligns data using time values, and plots ethograms for all matched IDs.
    Tracks F1 scores and Sensitivity (recall) for total, 'sefl', and 'control' conditions.

    Parameters:
    - freeze_frame: DataFrame containing the FreezeFrame data.
    - moseq_data: DataFrame containing the KPMS data.
    - plot_func: Function to plot the ethograms (default: plot_freeze_ethogram).
    - time_col: Column name representing the time axis in both DataFrames.
    """

    # Get unique cohort_id and day pairs
    matched_ids = moseq_data[['cohort_id', 'day']].drop_duplicates()

    # Initialize F1 and Sensitivity (recall) tracking
    f1_total = 0.0
    sensitivity_total = 0.0
    f1_count = 0
    f1_scores_by_condition = {'sefl': [], 'control': []}
    sensitivity_by_condition = {'sefl': [], 'control': []}

    # Loop through matched IDs and plot ethograms
    for _, row in matched_ids.iterrows():
        cohort_id, day = row['cohort_id'], row['day']

        # Filter data for the current cohort_id and day
        freeze_subset = freeze_frame[(freeze_frame['cohort_id'] == cohort_id) & (freeze_frame['day'] == day)].copy()
        moseq_subset = moseq_data[(moseq_data['cohort_id'] == cohort_id) & (moseq_data['day'] == day)].copy()

        # Skip if either subset is empty
        if freeze_subset.empty or moseq_subset.empty:
            print(f"Skipping plot for Cohort ID: {cohort_id}, Day: {day} due to empty data.")
            continue

        # Ensure time column exists in both DataFrames
        if time_col not in freeze_subset.columns or time_col not in moseq_subset.columns:
            print(f"Skipping Cohort ID: {cohort_id}, Day: {day} - Missing time column.")
            continue

        # Align based on time (truncate both datasets to the overlapping time range)
        min_time = max(freeze_subset[time_col].min(), moseq_subset[time_col].min())
        max_time = min(freeze_subset[time_col].max(), moseq_subset[time_col].max())

        freeze_subset = freeze_subset[(freeze_subset[time_col] >= min_time) & (freeze_subset[time_col] <= max_time)]
        moseq_subset = moseq_subset[(moseq_subset[time_col] >= min_time) & (moseq_subset[time_col] <= max_time)]

        # Ensure equal lengths by truncating
        min_length = min(len(freeze_subset), len(moseq_subset))
        freeze_subset = freeze_subset.iloc[:min_length]
        moseq_subset = moseq_subset.iloc[:min_length]

        # Plot using the provided function (default is plot_freeze_ethogram)
        try:
            plot_func(
                freeze_df=freeze_subset,
                moseq_df=moseq_subset,
                freeze_col='freeze',
                moseq_col='moseq_freeze',
                time_col=time_col,
                freeze_label='FreezeFrame',
                moseq_label='KPMS'
            )

            # Compute F1 score & Sensitivity (Recall)
            f1 = f1_score(freeze_subset['freeze'], moseq_subset['moseq_freeze'], average='binary')
            sensitivity = recall_score(freeze_subset['freeze'], moseq_subset['moseq_freeze'])

            f1_total += f1
            sensitivity_total += sensitivity
            f1_count += 1

            # Store F1 score & Sensitivity by condition
            condition = freeze_subset['condition'].iloc[0] if 'condition' in freeze_subset.columns else 'unknown'
            if condition in f1_scores_by_condition:
                f1_scores_by_condition[condition].append(f1)
                sensitivity_by_condition[condition].append(sensitivity)
            else:
                print(f"Unexpected condition: {condition}")

        except Exception as e:
            print(f"Error processing Cohort ID: {cohort_id}, Day: {day}. Skipping. {e}")
            continue

        print(f"Plotted ethogram for Cohort ID: {cohort_id}, Day: {day}, F1 Score: {f1:.2f}, Sensitivity: {sensitivity:.2f}")

    print("\nF1 & Sensitivity Scores Summary:")
    if f1_count > 0:
        print(f"Average F1 Score: {f1_total / f1_count:.2f}")
        print(f"Average Sensitivity: {sensitivity_total / f1_count:.2f}")


def calculate_syllable_freezing_proportion(moseq_df, freeze_df, freeze_col='freeze', syllable_col='syllable', time_col='time'):
    """
    Calculates the proportion of each syllable occurring within FreezeFrame-indicated freezing bouts.

    Parameters:
    - moseq_df: DataFrame containing KPMS data with syllable information and time column.
    - freeze_df: DataFrame containing FreezeFrame freezing state data with time column.
    - freeze_col: Column name in freeze_df indicating freezing state (1 = freezing, 0 = moving).
    - syllable_col: Column name in moseq_df representing syllable identities.
    - time_col: Column name representing the time axis in both DataFrames.

    Returns:
    - DataFrame summarizing syllable counts and their freezing association.
    """
    # Merge MoSeq and FreezeFrame data based on time
    merged_df = moseq_df[[syllable_col, time_col]].merge(
        freeze_df[[time_col, freeze_col]], on=time_col, how='left'
    )

    # Ensure freeze_col is binary (0 or 1), treating NaNs as non-freezing (0)
    merged_df[freeze_col] = merged_df[freeze_col].fillna(0).astype(int)

    # Group by syllable and compute counts
    syllable_counts = merged_df.groupby(syllable_col).size().reset_index(name='total_count')
    freezing_counts = merged_df.groupby(syllable_col)[freeze_col].sum().reset_index(name='freezing_count')

    # Merge results
    summary_df = pd.merge(syllable_counts, freezing_counts, on=syllable_col, how='left')
    summary_df['freezing_proportion'] = summary_df['freezing_count'] / summary_df['total_count']

    return summary_df

