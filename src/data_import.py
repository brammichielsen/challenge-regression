import pandas as pd
import glob

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

def find_csv_file(directory_path):
    """
    Find a .csv file in the specified directory.

    Parameters:
    directory_path (str): The path to the directory where the .csv files are located.

    Returns:
    str: The path to the .csv file if exactly one .csv file is found, otherwise None.
    """
    # Use glob to get the list of all .csv files in the directory
    csv_files = glob.glob(directory_path + "*.csv")

    # Check if there's exactly one .csv file in the directory
    if len(csv_files) == 1:
        # Get the first (and only) .csv file from the list
        file_path = csv_files[0]
        return file_path
    elif len(csv_files) > 1:
        print("Error: Multiple .csv files found in the directory.")
    else:
        print("Error: No .csv file found in the directory.")
    return None
