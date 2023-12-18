from models.base import Model

import keras
from keras import layers, optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from utility.paths import UtilityPath, DataPath
from utility.decorators import print_func_name


class GRU(Model):
    def __init__(self,
                 weight_path: str = "",
                 submission_path: str = "",
                 is_weight: bool = False,
                 seed: int = 42,
                 max_length: int = 128,
                 embedding_dim: int = 100):
        super().__init__(weight_path, submission_path, is_weight, seed)

        self.tokenizer = Tokenizer(oov_token="<unk>")
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.vocab_size = 0

        # Load weights
        self.model = keras.Sequential()

    def preprocessing(self, is_train: bool = True):
        steps = [
            "remove_ending",
            "remove_extra_space",
            "remove_space_around_emoji",
            "remove_extra_space",
            "reconstruct_emoji",
            "remove_extra_space",
            "emoji_to_tag",
            "reconstruct_last_emoji",
            "num_to_tag",
            "hashtag_to_tag",
            "repeat_symbols_to_tag",
            "elongate_to_tag",
            "remove_extra_space"
        ]

        if is_train:
            return ["drop_duplicates"] + steps

        return steps

    @print_func_name
    def update_vocabulary(self, x):
        self.tokenizer.fit_on_texts(x)
        self.vocab_size = len(self.tokenizer.word_index) + 2
        print(f"Vocabulary size: {self.vocab_size}")

    @print_func_name
    def padding(self, x):
        x_sequences = self.tokenizer.texts_to_sequences(x)
        x_pad = pad_sequences(x_sequences, maxlen=self.max_length, padding="post")
        return x_pad

    @print_func_name
    def generate_embedding_matrix(self):
        word_index = self.tokenizer.word_index

        embedding_index = {}

        with open(UtilityPath.GLOVES, "r") as f:
            for line in tqdm(f, desc="Loading GloVe"):
                word, coeff = line.split(maxsplit=1)
                coeff = np.fromstring(coeff, "f", sep=" ")
                embedding_index[word] = coeff

        print(f"Found {len(embedding_index)} word vectors")

        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        hit, miss = 0, 0
        for word, i in tqdm(word_index.items(), desc="Generating embedding matrix"):
            embedding_vector = embedding_index.get(word, None)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hit += 1
            else:
                miss += 1

        print(f"Converted {hit} words ({miss} missing)")

        return embedding_matrix

    @print_func_name
    def build_model(self, embedding_matrix):
        self.model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=self.max_length,
            mask_zero=True,
            trainable=False))

        self.model.add(layers.Bidirectional(
            layers.GRU(units=100, dropout=0.2, recurrent_dropout=0, activation="tanh",
                       recurrent_activation="sigmoid", unroll=False, use_bias=True,
                       reset_after=True)
        ))
        self.model.add(layers.Dense(100, activation="relu")),
        self.model.add(layers.Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(),
            metrics=["accuracy"])

        print("Model summary")
        print(self.model.summary())

    def train(self, x, y, batch_size: int = 128, epochs: int = 10):
        X_train, X_val, y_train, y_val = self.split_data(x, y)

        X_train_pad = self.padding(X_train)
        X_val_pad = self.padding(X_val)

        embedding_matrix = self.generate_embedding_matrix()
        self.build_model(embedding_matrix)

        print("Fitting model")
        self.model.fit(
            X_train_pad,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_pad, y_val))

        print("Saving model")
        self.model.save(f"{self.weight_path}/model", save_format='h5')

        print("Saving weights")
        self.model.save_weights(f"{self.weight_path}/model_weights.h5")

    def predict(self, x):
        if self.is_weight:
            self.model = keras.saving.load_model(f"{self.weight_path}/model")

        X_pad = self.padding(x)
        logits = self.model.predict(X_pad).squeeze()
        predictions = np.where(logits >= 0.5, 1, -1)

        self.submit(predictions)
