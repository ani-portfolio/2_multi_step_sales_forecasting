from datetime import datetime
from typing import Tuple

import pandas as pd

def train_test_split(df:pd.DataFrame, cutoff_date: datetime, target_column_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the data into training and test sets.

    Args:
        df (pd.DataFrame): The data to split.
        cutoff_date (datetime): The date to use as the cutoff for the split.
        target_column_name (str): The name of the column to use as the target.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: The training and test sets.
    """
    # Split the data into training and test sets
    df['date'] = pd.to_datetime(df['date'])
    train = df[df['date'] < cutoff_date].reset_index(drop=True)
    test = df[df['date'] >= cutoff_date].reset_index(drop=True)

    # Split the training and test sets into features and targets
    X_train = train.drop(columns=target_column_name)
    y_train = train[target_column_name]
    X_test = test.drop(columns=target_column_name)
    y_test = test[target_column_name]

    return X_train, y_train, X_test, y_test