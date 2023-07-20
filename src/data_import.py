import pandas as pd

def data_import(file_path):
    """
    Reads a dataset from a CSV file and returns it as a Pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: The dataset read from the CSV file.
    """
    data = pd.read_csv(file_path)
    return data