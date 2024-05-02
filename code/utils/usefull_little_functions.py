import pandas as pd

def get_all_ints_in_col(col):
    """
    Extracts integers from a column containing sets of integers in a DataFrame.

    Args:
    - col (pandas.DataFrame): The DataFrame containing the column.

    Returns:
    - list : A list containing all the integers extracted from the column.

    This function extracts all integers from a column in the DataFrame by exploding the column,
    dropping any NaN values, and converting the values to integers.
    """
    exploded = col.copy().explode().dropna()
    integers = exploded.astype(int)
    return list(integers)