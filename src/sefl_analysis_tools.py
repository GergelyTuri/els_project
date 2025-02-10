import pandas as pd
import numpy as np
import pingouin as pg
from pingouin import power_anova


def calculate_age_at_sefla(data):
    """
    Calculate 'age_at_sefla' and classify individuals as 'young' based on their age at 'sefla'.

    Parameters:
        data (pd.DataFrame): A DataFrame containing 'dob', 'date', 'day', 'cohort_id', and 'day' columns.

    Returns:
        pd.DataFrame: Updated DataFrame with 'age_at_sefla' and 'young' columns.
    """
    # Convert 'dob' and 'date' to datetime if they are not already
    data['dob'] = pd.to_datetime(data['dob'])
    data['date'] = pd.to_datetime(data['date'])
    
    # Initialize 'age_at_sefla' as None
    data['age_at_sefla'] = None

    # Iterate through the rows to calculate 'age_at_sefla'
    for idx, row in data.iterrows():
        if row['day'] == 'sefla':
            # Calculate age for 'sefla' day
            data.at[idx, 'age_at_sefla'] = (row['date'] - row['dob']).days / 7
        else:
            # Find the 'sefla' day for the same cohort
            sefla_row = data[(data['cohort_id'] == row['cohort_id']) & (data['day'] == 'sefla')]
            if not sefla_row.empty:
                data.at[idx, 'age_at_sefla'] = sefla_row.iloc[0]['age_at_sefla']

    # Convert 'age_at_sefla' to float
    data['age_at_sefla'] = data['age_at_sefla'].astype(float)

    # Create the 'young' column based on the 'age_at_sefla' value
    data['young'] = (data['age_at_sefla'] < 12).astype(str)

    return data


class AnalysisTools:
    def __init__(self, data):
        """
        Initialize the class with necessary data.
        """
        self.data = data

    def repeated_measures_anova(self, dv, within, between, subject, power=0.80, groups=2):
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

        self.data = self.data[(self.data['day'] != 'sefla') & (self.data['day'] != 'recall5')]

        # Perform the repeated measures ANOVA
        anova_result = pg.mixed_anova(
            data=self.data, dv=dv, within=within, between=between, subject=subject, effsize="np2"
        )
        
        # Extract eta squared from the ANOVA result
        eta_squared = anova_result["np2"].iloc[0]
        
        # Calculate the required sample size per group
        required_sample_size = power_anova(eta_squared=eta_squared, k=groups, power=power)
        
        return {
            "anova_result": anova_result,
            "required_sample_size": required_sample_size
        }

    def post_hoc_analysis_significant(self, dv, within, between, subject, p_adjust='bonferroni', alpha=0.05):
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

        self.data = self.data[(self.data['day'] != 'sefla') & (self.data['day'] != 'recall5')]

        # Perform pairwise t-tests
        pairwise_results = pg.pairwise_tests(
            data=self.data,
            dv=dv,
            between=between,
            within=within,
            subject=subject,
            padjust=p_adjust,
            effsize='cohen'
        )

        # Filter for significant results
        significant_results = pairwise_results[pairwise_results['p-corr'] < alpha]

        # Select relevant columns
        significant_results = significant_results[['Contrast', 'day', 'A', 'B', 'p-corr', 'cohen']]

        # Rename columns
        significant_results.rename(columns={'cohen': "Cohen'd"}, inplace=True)

        return significant_results