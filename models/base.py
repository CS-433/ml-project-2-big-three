from abc import ABC, abstractmethod
from time import strftime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Model(ABC):
    """
    Abstract class that models a generic neural network. Extended by GRU and BERT.
    """

    def __init__(self, weight_path: str, submission_path: str, is_weight: bool = False, seed: int = 42):
        """
        Init method for the abstract class.

        :param weight_path: weights path of the model. Model's parameters will be loaded and saved from this path.
        :type weight_path: str

        :param submission_path: submission path of the model. Model's predictions will be saved to this path.
        :type submission_path: str

        :param is_weight: whether to load weights or not.
        :type is_weight: bool. Default is False.

        :param seed: seed for reproducibility.
        :type seed: int. Default is 42.
        """

        self.weight_path = weight_path
        self.submission_path = submission_path
        self.is_weight = is_weight
        self.seed = seed

    @abstractmethod
    def preprocessing(self, is_train: bool = True):
        """
        Preprocessing methods to be used for different models

        :param is_train: whether to use train or test preprocessing methods
        :type is_train: bool. Default is True
        """

        pass

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int):
        """
        Train the model.

        :param x: input data (tweets)
        :type x: np.ndarray

        :param y:  labels
        :type y: np.ndarray

        :param batch_size: batch size for training
        :type batch_size: int

        :param epochs: epochs for training
        :type epochs: int
        """

        pass

    @abstractmethod
    def predict(self, x: np.ndarray):
        """
        Generate predictions.

        :param x: input data (tweets)
        :type x: np.ndarray
        """
        
        pass

    def split_data(self, x: np.ndarray, y: np.ndarray, test_size: float = 0.2, shuffle: bool = True, **kwargs) -> tuple:
        """
        Split data into train and validation sets.
        :param x: input data (tweets)
        :type x: np.ndarray

        :param y: labels
        :type y: np.ndarray

        :param test_size: size of the validation set
        :type test_size: float. Default is 0.2

        :param shuffle: whether to shuffle the data or not
        :type shuffle: bool. Default is True

        :param kwargs: additional arguments
        :type kwargs: dict

        :return: train and validation sets
        :rtype: tuple
        """

        return train_test_split(x, y, test_size=test_size, shuffle=shuffle, random_state=self.seed, **kwargs)

    def submit(self, predictions: np.ndarray | list):
        """
        Create a submission file for AICrowd.
        :param predictions: generated predictions
        :type predictions: np.ndarray | list
        """

        submission = pd.DataFrame({"Id": np.arange(1, len(predictions) + 1), "Prediction": predictions})
        submission["Prediction"] = submission["Prediction"].replace(0, -1)
        submission = submission.astype(int)

        submission.to_csv(f"{self.submission_path}/submission_{strftime('%Y-%m-%d_%H:%M:%S')}.csv", index=False)
