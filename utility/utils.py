import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from utility.paths import DataPath
import gensim


def merge_dataframe(arr_pos: list, arr_neg: list) -> pd.DataFrame:
    """
    Merge the two lists into a single DataFrame

    :param arr_pos: Positive tweets
    :type arr_pos: list

    :param arr_neg: Negative tweets
    :type arr_neg: list

    :return: DataFrame with the tweets and their labels
    :rtype: pd.DataFrame
    """
    # Create a DataFrame from arr_pos with "value" column set to 1
    df_pos = pd.DataFrame({'text': arr_pos, 'label': 1})

    # Create a DataFrame from arr_neg with "value" column set to -1
    df_neg = pd.DataFrame({'text': arr_neg, 'label': 0})

    # Concatenate the two DataFrames
    result_df = pd.concat([df_pos, df_neg], ignore_index=True)

    return result_df


def load_data(path_neg_dataset: str = DataPath.TRAIN_NEG, path_pos_dataset: str = DataPath.TRAIN_POS) -> pd.DataFrame:
    """
    Load the data from the files and return them as lists

    :param path_neg_dataset: Path to the negative dataset
    :type path_neg_dataset: str

    :param path_pos_dataset: Path to the positive dataset
    :type path_pos_dataset: str

    :return: DataFrame with the tweets and their labels
    :rtype: pd.DataFrame
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


def load_full_data(path_neg_dataset: str = DataPath.TRAIN_NEG_FULL,
                   path_pos_dataset: str = DataPath.TRAIN_POS_FULL) -> pd.DataFrame:
    """
    Load the full data from the files and return them as lists

    :param path_neg_dataset: Path of full negative dataset
    :type path_neg_dataset: str

    :param path_pos_dataset: Path of full positive dataset
    :type path_pos_dataset: str

    :return: DataFrame with the tweets and their labels
    :rtype: pd.DataFrame
    """
    return load_data(path_neg_dataset, path_pos_dataset)


def load_test_data(path_test_dataset: str = DataPath.TEST) -> pd.DataFrame:
    """
    Load the test data from the file and return them as lists

    :param path_test_dataset: Path of test dataset
    :type path_test_dataset: str

    :return: DataFrame with the tweets
    :rtype: pd.DataFrame
    """
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


def load_preprocessed_data() -> pd.DataFrame:
    """
    Load cropped preprocessed data from the file

    :return: DataFrame with the tweets and their labels
    :rtype: pd.DataFrame
    """

    # Load the dataset
    train_dataset = pd.read_csv(DataPath.ML_TRAIN_CROPPED)
    train_dataset = train_dataset.drop_duplicates()

    return train_dataset


def load_preprocessed_full_data() -> pd.DataFrame:
    """
    Load full preprocessed data from the file

    :return: DataFrame with the tweets and their labels
    :rtype: pd.DataFrame
    """

    # Load the dataset
    train_dataset = pd.read_csv(DataPath.ML_TRAIN)
    train_dataset = train_dataset.drop_duplicates()

    return train_dataset


def load_preprocessed_test_data() -> pd.DataFrame:
    """
    Load preprocessed test data from the file

    :return: DataFrame with the tweets
    :rtype: pd.DataFrame
    """
    # Load the dataset
    test_dataset = pd.read_csv(DataPath.ML_TEST)

    return test_dataset


def check_parentheses(s: str) -> bool:
    """
    Check if the parentheses are balanced
    :param s: String to check
    :type s: str

    :return: True if the parentheses are balanced, False otherwise
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


def remove_parentheses(s: str) -> str:
    """
    Remove parentheses from a string

    :param s: String to remove parentheses from
    :type s: str

    :return: String without parentheses
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
                result += s[i + 1:j]
                i = j + 1
            else:
                result += s[i]
                i += 1
        else:
            result += s[i]
            i += 1
    return result


def export_submission(y_test_pred: np.ndarray, export_path_and_name: str = 'test_predictions.csv'):
    """
    Exports test predictions to provided file path in the correct AICrowd format.
    :param y_test_pred: Array of predictions of shape (10000, 1)
    :type y_test_pred: np.ndarray

    :param export_path_and_name: File path and name to export predictions to.
    :type export_path_and_name: str
    """

    # Add Check to make sure that the length of y_test_pred is correct
    if y_test_pred.shape != (10000,):
        print(f"Error: Shape of y_test_pred is {y_test_pred.shape} instead of (10000,)")
        return None

    # Create array of indexes
    indexes = np.arange(1, 10001)

    # Check if the labels are binary and 0 and 1
    if np.array_equal(np.unique(y_test_pred), np.array([0, 1])):
        y_test_pred_2 = np.where(y_test_pred == 0, -1, y_test_pred)
    elif np.array_equal(np.unique(y_test_pred), np.array([-1, 1])):
        y_test_pred_2 = y_test_pred
    else:
        print("Error: Labels are not binary and 0 and 1 or -1 and 1")
        return None
    submission = pd.DataFrame({'Id': indexes, 'Prediction': y_test_pred_2})
    submission.to_csv(export_path_and_name, index=False)


class WordEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Vectorizer that averages the word vectors of all words in a document.
    """

    def __init__(self, word_vectors: gensim.models.keyedvectors.Word2VecKeyedVectors):
        """
        :param word_vectors: Word vectors
        :type word_vectors: gensim.models.keyedvectors.Word2VecKeyedVectors
        """
        self.word_vectors = word_vectors
        self.dim = word_vectors.vector_size

    def transform(self, x: list):
        """
        Transform a list of documents to a list of document vectors.
        :param x: List of documents
        :type x: list

        :return: List of document vectors
        :rtype: np.ndarray
        """
        return np.array([
            np.mean([self.word_vectors[w] for w in words if w in self.word_vectors]
                    or [np.zeros(self.dim)], axis=0)
            for words in x
        ])
