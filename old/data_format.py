from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def train_test_split_data(data, target_col='Price of property in euro', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    data (DataFrame): The input dataset.
    target_col (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Seed for the random number generator.

    Returns:
    X_train (DataFrame): The features of the training set.
    X_test (DataFrame): The features of the testing set.
    y_train (Series): The target values of the training set.
    y_test (Series): The target values of the testing set.
    """
    
    # Separate the target variable (Price of property in euro) from the features
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Perform the Train-Test split with the specified ratio and random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Apply Min-Max scaling to the dataset.

    Parameters:
    X_train (DataFrame): The features of the training set.
    X_test (DataFrame): The features of the testing set.

    Returns:
    X_train_scaled (DataFrame): The scaled features of the training set.
    X_test_scaled (DataFrame): The scaled features of the testing set.
    """
   
    scaler = MinMaxScaler()
    # Apply Min-Max scaling to the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply Min-Max scaling to the testing data
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
