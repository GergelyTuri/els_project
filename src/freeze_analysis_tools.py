import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def find_freeze_transitions(freeze_frame_data):
    '''
    Identifies the timestamps where the 'freeze' in the freezeframe transitions: 
    - From 1 to 0 (offset, end of a freeze period)
    - From 0 to 1 (onset, start of a freeze period)

    Parameters:
    - freeze_frame_data: freeze frame DataFrame with a 'freeze' column

    Returns:
    - A DataFrame with columns ['t(sec)', 'transition_type', 'cohort_id', 'day'],
      sorted by 'cohort_id', 'day', and 't(sec)' (if available).
    '''

    if 'freeze' not in freeze_frame_data.columns: 
        raise ValueError('Input DataFrame does not contain a "freeze" column')
    
    df = freeze_frame_data.copy()
    
    # Shift freeze values to detect transitions
    df['freeze_shifted'] = df['freeze'].shift(1, fill_value=0)

    # Find transition points
    offset_transitions = df[(df['freeze_shifted'] == 1) & (df['freeze'] == 0)][['time']].copy()
    offset_transitions['transition_type'] = 'offset'

    onset_transitions = df[(df['freeze_shifted'] == 0) & (df['freeze'] == 1)][['time']].copy()
    onset_transitions['transition_type'] = 'onset'

    # Ensure 'cohort_id' and 'day' are preserved properly
    for col in ['cohort_id', 'day']:
        if col in freeze_frame_data.columns:
            offset_transitions[col] = df.loc[offset_transitions.index, col].values
            onset_transitions[col] = df.loc[onset_transitions.index, col].values
        else:
            offset_transitions[col] = None
            onset_transitions[col] = None

    # Combine transitions
    all_transitions = pd.concat([offset_transitions, onset_transitions], ignore_index=True)

    # Sort by 'cohort_id', 'day', and 't(sec)' if they exist
    sort_columns = [col for col in ['cohort_id', 'day', 'time'] if col in all_transitions.columns]
    all_transitions = all_transitions.sort_values(by=sort_columns).reset_index(drop=True)

    return all_transitions


def calculate_median_freeze_duration(freeze_frame_data):
    '''
    Calculates the overall median freeze duration across all mice and day

    Parameters:
    - freeze_frame_data: DataFrame, containing columns: 
        - 't(sec)': time in seconds
        - 'freeze': binary column indicating whether the mouse is freezing
        - 'cohort)id': id for individual mouse
        - 'day': day number for the experiment

    Returns:
    - A float, the median freeze duration across all mice and day combinations
    '''

    required_columns = ['t(sec)', 'freeze', 'cohort_id', 'day']
    if not required_columns.issubset(freeze_frame_data.columns): 
        raise ValueError(f'Input Dataframe does not contain all required columns {required_columns}')
    
    cohort_ids = freeze_frame_data['cohort_id'].unique()
    days = freeze_frame_data['day'].unique()

    all_freeze_durations = []

    for cohort_id in cohort_ids:
        for day in days:
            cohort_day_data = freeze_frame_data[
                (freeze_frame_data['cohort_id'] == cohort_id) &
                (freeze_frame_data['day'] == day)]
            
            if cohort_day_data.empty:
                print(f"No data found for cohort_id: {cohort_id}, day: {day}")
                continue 
            
            freeze_durations = []
            is_freezing = False
            
            for _, row in cohort_day_data.iterrows():
                if row['freeze'] == 1 and not is_freezing:
                    is_freezing = True
                    freeze_start = row['t(sec)']
                elif row['freeze'] == 0 and is_freezing:
                    is_freezing = False
                    freeze_end = row['t(sec)']
                    freeze_durations.append(freeze_end - freeze_start)
            
            if is_freezing and freeze_start is not None: 
                freeze_end = cohort_day_data['t(sec)'].iloc[-1]
                freeze_durations.append(freeze_end - freeze_start)

            all_freeze_durations.extend(freeze_durations)

    overall_med = np.median(all_freeze_durations) if all_freeze_durations else 0
    return overall_med

import pandas as pd

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
    # Columns we want to preserve if present:
    preserve_cols = ['condition', 'sex', 'young', 'age']  # Adjust as needed
    
    bouts = []
    
    # Group by each trial (cohort_id, day)
    grouped = freeze_frame_data.groupby(['cohort_id', 'day'], as_index=False)
    
    for (cohort, day), group in grouped:
        group = group.sort_values('t(sec)')
        
        # Extract any extra metadata from the first row of this group
        # (assuming these do not change within the same (cohort_id, day) session)
        extra_data = {}
        for col in preserve_cols:
            if col in group.columns:
                extra_data[col] = group[col].iloc[0]
        
        trial_start = group['t(sec)'].iloc[0]
        trial_end   = group['t(sec)'].iloc[-1]
        
        in_bout    = False
        bout_start = None
        
        # Detect freezing bouts
        for _, row in group.iterrows():
            if row['freeze'] == 1 and not in_bout:
                # Start of a new bout
                in_bout    = True
                bout_start = row['t(sec)']
            elif row['freeze'] == 0 and in_bout:
                # End of the current bout
                bout_end = row['t(sec)']
                bouts.append({
                    'cohort_id': cohort,
                    'day': day,
                    'bout_start': bout_start,
                    'bout_end': bout_end,
                    'duration': bout_end - bout_start,
                    **extra_data  # Include preserved metadata
                })
                in_bout    = False
                bout_start = None
        
        # If still in a bout at the end of the trial, close it
        if in_bout:
            bout_end = trial_end
            bouts.append({
                'cohort_id': cohort,
                'day': day,
                'bout_start': bout_start,
                'bout_end': bout_end,
                'duration': bout_end - bout_start,
                **extra_data
            })
    
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
    bouts_df['minute_bin'] = (bouts_df['bout_start'] // 60 + 1).astype(int)

    # Only consider bouts that start within the experiment time
    max_minute = total_experiment_time // 60
    bouts_df = bouts_df[bouts_df['minute_bin'] <= max_minute]

    # Calculate summary statistics for each minute bin
    summary = {}
    for minute in range(1, max_minute + 1):
        minute_data = bouts_df[bouts_df['minute_bin'] == minute]
        summary[f"minute_{minute}"] = {
            'bout_count': len(minute_data),
            'median_duration': minute_data['duration'].median() if not minute_data.empty else None,
            'mean_duration': minute_data['duration'].mean() if not minute_data.empty else None
        }
    
    return bouts_df, summary



def plot_freeze_duration_pointplot(bouts_df, group_cols, subset_col):
    """
    Creates a side-by-side pointplot of 'duration' by 'minute_bin',
    subsetting the data by a given column while using interaction groups for hue.

    Parameters
    ----------
    bouts_df : pd.DataFrame
        The dataset containing:
        - 'minute_bin': numeric time bins
        - 'duration': freezing duration values
        - plus any columns to be used for grouping.
    group_cols : list of str
        List of column names to combine into a single interaction variable for hue.
    subset_col : str
        Column used to subset the data into two separate plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : list of matplotlib.axes.Axes
        The two subplot axes.
    """
    # Ensure 'minute_bin' is numeric
    df = bouts_df.copy()
    df['minute_bin'] = df['minute_bin'].astype(int)

    # Unique values in subset column (we assume exactly 2 unique values)
    unique_values = df[subset_col].dropna().unique()
    if len(unique_values) != 2:
        raise ValueError(f"Expected exactly 2 unique values in '{subset_col}', but found {len(unique_values)}")

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)

    # Loop through each subset value and create a pointplot
    for ax, subset_value in zip(axes, unique_values):
        subset_df = df[df[subset_col] == subset_value]

        # Create an interaction column for hue
        subset_df.loc[:, 'interaction'] = subset_df[group_cols[0]].astype(str)
        for col in group_cols[1:]:
            subset_df.loc[:, 'interaction'] += "_" + subset_df[col].astype(str)

        # Ensure legend labels are sorted by the first word, then the second
        sorted_keys = sorted(
            subset_df['interaction'].unique(),
            key=lambda k: (k.split("_")[0].lower(), k.split("_")[1].lower())  # Sort first by first word, then second
        )

        legend_labels = {
            key: " ".join(word.capitalize() for word in key.replace("_", " ").split())
            for key in sorted_keys
        }

        # Define a color palette based on sorted_keys
        palette = sns.color_palette("deep", n_colors=len(sorted_keys))

        # Sort minute bins for consistent x-axis order
        bins_sorted = sorted(subset_df['minute_bin'].unique())

        # Plot
        sns.pointplot(
            data=subset_df,
            x='minute_bin',
            y='duration',
            hue='interaction',
            hue_order=sorted_keys,  # Ensures hue follows sorted order
            order=bins_sorted,
            dodge=True,
            ax=ax,
            palette=palette
        )

        # Set subplot title and labels
        ax.set_title(f"{subset_col.capitalize()}: {subset_value}", fontsize=14)
        ax.set_xlabel("Minute Bin", fontsize=12)
        ax.set_ylabel("Freezing Duration (s)", fontsize=12)

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [legend_labels[label] for label in labels]
        ax.legend(handles, new_labels, title="Condition", loc="upper right", fontsize=10, title_fontsize=11)

    # Improve aesthetics
    sns.despine()
    plt.tight_layout()

    return fig, axes
