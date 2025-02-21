import pandas as pd 
import numpy as np

import pandas as pd

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

