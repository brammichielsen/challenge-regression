import pandas as pd

def data_import(file_path):
    """
    Reads a CSV file and returns the data as a Pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    DataFrame: The data read from the CSV file.
    """
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # Return the data
    return data