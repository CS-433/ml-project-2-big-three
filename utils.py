import numpy as np
import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from random import shuffle
from collections import Counter
import string

def merge_dataframe(arr_pos, arr_neg):
    # Create a DataFrame from arr_pos with "value" column set to 1
    df_pos = pd.DataFrame({'text': arr_pos, 'label': 1})
    
    # Create a DataFrame from arr_neg with "value" column set to -1
    df_neg = pd.DataFrame({'text': arr_neg, 'label': 0})
    
    # Concatenate the two DataFrames
    result_df = pd.concat([df_pos, df_neg], ignore_index=True)
    
    return result_df

def load_data(path_neg_dataset='twitter-datasets/train_neg.txt', path_pos_dataset='twitter-datasets/train_pos.txt'):
    """
    Load the data from the files and return them as lists
    :return: train_neg, train_pos
    """
    # Initialize empty lists to store the lines from the files
    train_neg = []
    train_pos = []

    # File paths for the datasets
    train_neg_file = path_neg_dataset
    train_pos_file = path_pos_dataset

    # Read lines from 'train_neg' dataset and store them in train_neg_lines
    with open(train_neg_file, 'r', encoding='utf-8') as file:
        for line in file:
            train_neg.append(line.strip())  # Remove newline characters

    # Read lines from 'train_pos' dataset and store them in train_pos_lines
    with open(train_pos_file, 'r', encoding='utf-8') as file:
        for line in file:
            train_pos.append(line.strip())  # Remove newline characters


    return merge_dataframe(train_pos, train_neg)

def load_full_data(path_neg_dataset='twitter-datasets/train_neg_full.txt', path_pos_dataset='twitter-datasets/train_pos_full.txt'):
    return load_data(path_neg_dataset, path_pos_dataset)


def load_preprocessed_data(path_train_dataset='twitter-datasets/train_preprocessed_full.txt', header=True):
    # Initialize empty lists to store the lines from the files
    train = []
    labels = []

    # File paths for the datasets
    train_file = path_train_dataset

    # Read lines from 'train_neg' dataset and store them in train_neg_lines
    with open(train_file, 'r', encoding='utf-8') as file:
        if header:
            next(file)  # Skip the header line
        for line in file:
            data = line.strip().split(',')
            train.append(data[0])  # First column as text
            labels.append(data[1])  # Second column as label

    df_train = pd.DataFrame({'text': train, 'label': labels})

    return df_train

def load_test_data(path_test_dataset='twitter-datasets/test_data.txt'):
    # Initialize empty lists to store the lines from the files
    test = []

    # File paths for the datasets
    test_file = path_test_dataset

    # Read lines from 'train_neg' dataset and store them in train_neg_lines
    with open(test_file, 'r', encoding='utf-8') as file:
        for line in file:
            test.append(line.strip())  # Remove newline characters

    df_test = pd.DataFrame({'text': test})

    return df_test

    

def check_parentheses(s):
    """
    :type s: str
    :rtype: bool
    """
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            if count == 0:
                return False
            count -= 1
    return count == 0

def remove_parentheses(s):
    """
    :type s: str
    :rtype: str
    """
    result = ""
    i = 0
    while i < len(s):
        if s[i] == "(":
            j = i
            while j < len(s) and s[j] != ")":
                if s[j] == "(":
                    result += s[i:j]
                    i = j
                j += 1
            if j < len(s) and s[j] == ")":
                result += s[i+1:j]
                i = j + 1
            else:
                result += s[i]
                i += 1
        else:
            result += s[i]
            i += 1
    return result



def export_submission(y_test_pred, export_path_and_name='test_predictions.csv'):
    """
    Exports test predictions to provided file path in the correct AICrowd format.
    Args:
        y_test_pred (np.array): Array of predictions of shape (10000, 1)
        export_path_and_name (str): File path and name to export predictions to.
    Returns:
        None
    """
    # Add Check to make sure that the length of y_test_pred is correct
    if y_test_pred.shape != (10000,):
        print('Error: Shape of y_test_pred is {} instead of (10000,)'.format(y_test_pred.shape))
        return None
    # Create array of indexes
    indexes = np.arange(1, 10001)  
    # Check if the labels are binary and 0 and 1
    if (np.array_equal(np.unique(y_test_pred), np.array([0, 1])) == True):
        y_test_pred_2 = np.where(y_test_pred == 0, -1, y_test_pred)
    elif (np.array_equal(np.unique(y_test_pred), np.array([-1, 1])) == True):
        y_test_pred_2 = y_test_pred
    else:
        print('Error: Labels are not binary and 0 and 1 or -1 and 1')
        return None
    submission = pd.DataFrame({'Id':indexes,'Prediction': y_test_pred_2})
    submission.to_csv(export_path_and_name, index=False)