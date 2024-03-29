"""
Helper functions for loading data from Google Drive and Google Sheets.
Requires review!
"""

import gspread
import pandas as pd
from google.auth import default
from google.colab import auth, drive


def g_drive():
    """mounts Google Drive"""
    drive.mount("/content/drive")
    return


def user_auth():
    """authenticates user"""
    auth.authenticate_user()
    creds, _ = default()
    gspread.authorize(creds)
    return


def load_spreadsheet_data(spreadsheet, sheet):
    """
    Loads data from a specified sheet in a given Google Spreadsheet.

    Parameters:
    spreadsheet (str): The name of the Google Spreadsheet.
    sheet (str): The name of the sheet in the Spreadsheet.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the specified sheet.
    """
    workbook = gc.open(spreadsheet)
    values = workbook.worksheet(sheet).get_all_values()
    df = pd.DataFrame.from_records(values[:]).drop(0).reset_index(drop=True)
    df.columns = values[0]
    return df
