import pandas as pd
import random
import torch
import numpy as np
from tqdm import tqdm

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

    random.shuffle(l)

    if len(l)==0:
        print("No family negatives disponible - probably looking for root's children. Using random negatives.")
        l = random.sample(range(1, 40000), num_el)
    elif len(l) == num_el:
        l = l
    elif len(l) > num_el:
        l = l[:num_el]  
    elif len(l) < num_el:
        extra_elements = random.choices(l, k = num_el - len(l))
        l = l + extra_elements
    
    return l

def make_family_negatives(family_df, neg_group_col: str, num_neg_per_pos):
    '''
    Generates negative examples for each positive example in a DataFrame.

    For each positive example in the DataFrame `family_df`, it creates negative examples by adjusting the number of elements in the specified column `neg_group_col`.

    Parameters:
        family_df (DataFrame): The input DataFrame containing positive examples.
        neg_group_col (str) : The column name representing groups of elements.
        num_neg_per_pos (int) : The desired number of negative examples per positive example.

    Returns:
        DataFrame: A DataFrame with negative examples generated for each positive example.
    '''
    if neg_group_col not in family_df.columns:
        raise KeyError(f'{neg_group_col} not in {list(family_df.columns)}')
                          
    print(num_neg_per_pos)
    tqdm.pandas(desc=f'Making family negatives')
    a = family_df.progress_apply(lambda row : torch.Tensor(correct_numel(list(row[neg_group_col]),num_el = num_neg_per_pos)), axis =1)
    display(len(a[0]))

    return a

def make_source_family_neg_dict(input_df, source_col = 'source', neg_col = 'family', num_neg_per_pos=224):

    '''
    Generates a dictionary with negative examples for each positive example based on a source column in an input DataFrame.

    Creates a dictionary where each key is a positive example from the source column, and each value is a list of negative examples generated from the corresponding positive example.

    Parameters:
        input_df (DataFrame): The input DataFrame containing positive and negative examples.
        source_col (str): The column name representing the source of positive examples.
        neg_col (str): The column name representing groups of elements.
        num_neg_per_pos (int): The desired number of negative examples per positive example.

    Returns:
        dict: A dictionary mapping positive examples to lists of negative examples.
    '''
    df = input_df[[source_col, neg_col]].copy()
    if not isinstance(df[source_col][0], np.int64):
        df[source_col] = df.apply(lambda row: int(list(row[source_col])[0]), axis = 1)

    if not isinstance(df[neg_col][0], torch.Tensor):
        df[neg_col]    = df.apply(lambda row: torch.Tensor(list(row[neg_col])), axis = 1)
    
    df['numeled_neg_col'] = make_family_negatives(df, neg_col, num_neg_per_pos)

    return df.set_index(source_col)['numeled_neg_col'].to_dict()

def try_to_associate_family_neg(id, family_neg_dict, num_neg):
    try :
        return family_neg_dict[id]
    except :
        print(f'node not found in dict ! {id}')
        return torch.tensor(random.sample(range(1, 40000), num_neg))
    
def make_family_neg_for_batch(batch, family_neg_dict):
    
    num_neg = batch['go']['dst_neg_index'].size()[1]
    pos = batch['go']['dst_pos_index'].numpy()
    df = pd.DataFrame(pos, columns=['pos'])
    df['family_neg'] = df.apply(lambda row : try_to_associate_family_neg(id = row['pos'], family_neg_dict=family_neg_dict, num_neg=num_neg), axis =1)
    return torch.stack(list(df['family_neg'])).type(dtype=torch.LongTensor)