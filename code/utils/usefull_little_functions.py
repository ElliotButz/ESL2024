import pandas as pd
import random
import torch

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

def make_dict_from_columns(df, col_keys, col_values, tensorize_values = False):
    """
    Creates a dictionary from specified columns of a DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame from which to create the dictionary.
    - col_keys (str or list): Column name(s) to use as keys in the dictionary.
    - col_values (str or list): Column name(s) to use as values in the dictionary.

    Returns:
    - result_dict (dict): Dictionary created from specified columns.
    """
    if not tensorize_values :
        return df.set_index(col_keys)[col_values].to_dict()
    else :
        df = df.set_index(col_keys)[col_values]
        df[col_values].apply(lambda row : row[col_values])

def correct_numel(l, num_el):

    '''
    Correct the number of elements in a list by adjusting its length.

    If the length of `l` matches `num_el`, the function returns `l` unchanged.
    If `l` is longer than `num_el`, it returns a sublist containing the first `num_el` elements.
    If `l` is shorter than `num_el`, it randomly selects elements from `l` to append to the end until the desired length is reached.

    Parameters:
        l (list)     : The input list.
        num_el (int) : The desired number of elements in the list.

    Returns:
        list: A list with the corrected number of elements.
    '''

    list(l)
    random.shuffle(l)
    if len(l) == num_el:
        return l
    elif len(l) > num_el:
        return l[:num_el]  
    elif len(l) < num_el:
        extra_elements = random.choices(l, k = num_el - len(l))
        l = l + extra_elements
    return l

def make_source_family_neg_dict(df, source_col = 'source', neg_col = )
    serie = df.apply(lambda row: torch.Tensor(list(row['family_without_lineage'])), axis = 1)
