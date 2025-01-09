import pandas as pd
import numpy as np
import pingouin as pg
from pingouin import power_anova

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
