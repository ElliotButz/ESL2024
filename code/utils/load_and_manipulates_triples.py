import pykeen
import pandas as pd
import os
import numpy as np
import constants as c

def triples_from_csv(path_to_csv, columns_to_use = c.Node.features.value):
    """
    Creates triples from a CSV.

    Parameters:
    - path_to_csv (str): Filepath or buffer to the input CSV file.
    - columns_to_use (list): A list of the columns to consider.

    Returns:
    - triples (pandas.DataFrame): Output DataFrame in triple format.
                                  Subjects are index items, predicates are column names from columns_to_use, objects are non-NaN values in columns.
    """

    df = pd.read_csv(filepath_or_buffer=path_to_csv, sep = ',', index_col = 0)
    df.columns = df.columns.str.lower()
    
    # Create a list of triples
    triples = []
    # Drop feature columns
    columns_to_drop = [col for col in columns_to_use if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    df = df.replace({np.nan:None})
    
    for index, row in df.iterrows():
        for column in df.columns:
            if row[column] is not None:
                for predicate in row[column].split('|'):
                    triples.append([index, column, predicate])

    # Create a dataframe from the list of triples
    return pd.DataFrame(triples, columns=['subject', 'predicate', 'object'])


def tsv_from_csv(output_path, path_to_csv, columns_to_use = c.Node.features.value):
    """
    Creates .tsv file of triples from a CSV.
    Subjects are index items, predicates are column names from columns_to_use, objects are non-NaN values in columns.

    Parameters:
    - output_path (str): Filepath or buffer to the output TSV file.
    - path_to_csv (str): Filepath or buffer to the input CSV file.
    - columns_to_use (list): A list of the columns to consider.
    
    Returns:
    - None in triple format.
    """

    triples_from_csv(path_to_csv,
                     columns_to_use = columns_to_use).to_csv(path_or_buf=output_path,
                                                             sep = '\t', index=False)