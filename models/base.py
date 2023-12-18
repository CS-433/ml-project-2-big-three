from abc import ABC, abstractmethod
from time import strftime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Model(ABC):
    def __init__(self, weight_path: str, submission_path: str, is_weight: bool = False, seed: int = 42):
        self.weight_path = weight_path
        self.submission_path = submission_path
        self.is_weight = is_weight
        self.seed = seed

    @abstractmethod
    def train(self, x, y, batch_size: int, epochs: int):
        pass

    @abstractmethod
    def preprocessing(self, is_train: bool = True):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def split_data(self, x, y, test_size: float = 0.2, shuffle: bool = True, **kwargs):
        return train_test_split(x, y, test_size=test_size, shuffle=shuffle, random_state=self.seed, **kwargs)

    def submit(self, predictions: np.ndarray | list):
        submission = pd.DataFrame({"Id": np.arange(1, len(predictions) + 1), "Prediction": predictions})
        submission["Prediction"] = submission["Prediction"].replace(0, -1)
        submission = submission.astype(int)

        submission.to_csv(f"{self.submission_path}/submission_{strftime('%Y-%m-%d_%H:%M:%S')}.csv", index=False)
