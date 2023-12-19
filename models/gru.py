from models.base import Model

import keras
from keras import layers, optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant

import numpy as np
from tqdm.auto import tqdm

from utility.paths import UtilityPath
from utility.decorators import print_func_name


class GRU(Model):
    """
    A GRU-based model for tweet classification task. The model is pretrained on GloVe embeddings, provided by [Stanford
    NLP Lab](https://nlp.stanford.edu/projects/glove/).
    """

    def __init__(self,
                 weight_path: str,
                 submission_path: str,
                 is_weight: bool = False,
                 seed: int = 42,
                 max_length: int = 128,
                 embedding_dim: int = 100):
        """
        Initialize the GRU model with specified parameters.

        :param weight_path: Path to the pre-trained weights.
        :type weight_path: str

        :param submission_path: Path for saving submissions.
        :type submission_path: str

        :param is_weight: Flag to indicate if pre-trained weights are used.
        :type is_weight: bool. Default to False.

        :param seed: Random seed for reproducibility.
        :type seed: int. Default to 42.

        :param max_length: Maximum length of input sequences.
        :type max_length: int. Default to 128.

        :param embedding_dim: Dimension of the embedding vectors.
        :type embedding_dim: int. Default to 100.
        """

        super().__init__(weight_path, submission_path, is_weight, seed)

        self.tokenizer = Tokenizer(oov_token="<unk>")
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.vocab_size = 0

        # Load weights
        self.model = keras.Sequential()

    def preprocessing(self, is_train: bool = True) -> list[str]:
        """
        Preprocess method for the input data.

        This method applies a series of text processing steps to prepare the data for the model.
        The steps differ slightly based on whether the data is for training or inference.

        :param is_train: Flag to indicate if the data is for training.
        :type is_train: bool. Default to True.

        :return: A list of preprocessing steps to be applied.
        :rtype: list[str]
        """
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
    def update_vocabulary(self, x: np.ndarray):
        """
        Update the vocabulary of the tokenizer.

        :param x: Input data of tweet.
        :type x: np.ndarray
        """

        # Updates internal vocabulary based on a list of texts
        self.tokenizer.fit_on_texts(x)
        self.vocab_size = len(self.tokenizer.word_index) + 2  # +2 for <pad> and <unk>
        print(f"Vocabulary size: {self.vocab_size}")

    @print_func_name
    def padding(self, x: np.ndarray) -> np.ndarray:
        """
        Pad the input sequences to the same length.

        Each word in a tweet is replaced with its vocabulary index in a bag-of-words style,
        padding tweets to a maximum of `self.max_length` (default to 128) words using '0' as the padding character.
        :param x: Input data of tweet.
        :type x: np.ndarray

        :return: Matrix with shape (len(x), self.max_length)
        :rtype: np.ndarray
        """
        # Transforms each text in texts to a sequence of integers
        x_sequences = self.tokenizer.texts_to_sequences(x)
        x_pad = pad_sequences(x_sequences, maxlen=self.max_length, padding="post")  # post: pad after each sequence
        return x_pad

    @print_func_name
    def generate_embedding_matrix(self) -> np.ndarray:
        """
        Creates a word embedding matrix based on the vocabulary words,
        representing each word as a vector of size `self.embedding_dim`.
        Embeddings are derived from GloVe pre-trained on Twitter data (https://nlp.stanford.edu/projects/glove/),
        considering only those vocabulary words present in the pre-trained model.

        :return: The embedding matrix, where each row aligns with a word from the vocabulary, is indexed according to
        the word's position in the vocabulary.
        :rtype: np.ndarray
        """
        # Get word index from tokenizer
        word_index = self.tokenizer.word_index

        embedding_index = {}

        # Load GloVe embeddings
        with open(UtilityPath.GLOVES, "r") as f:
            for line in tqdm(f, desc="Loading GloVe"):
                word, coeff = line.split(maxsplit=1)
                coeff = np.fromstring(coeff, "f", sep=" ")
                embedding_index[word] = coeff

        print(f"Found {len(embedding_index)} word vectors")

        # Create embedding matrix
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        hit, miss = 0, 0
        for word, i in tqdm(word_index.items(), desc="Generating embedding matrix"):
            embedding_vector = embedding_index.get(word, None)
            # Words not found in embedding index will be all-zeros
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hit += 1
            else:
                miss += 1

        print(f"Converted {hit} words ({miss} missing)")

        return embedding_matrix

    @print_func_name
    def build_model(self, embedding_matrix):
        """
        Build and compile GRU model.

        :param embedding_matrix:
        :type embedding_matrix: np.ndarray
        """

        self.model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=self.max_length,
            mask_zero=True,  # Special character so set to True
            trainable=False))  # Freeze embedding layer since we are using pre-trained weights

        # Using 2 dropouts: 1 for GRU and 1 for RNN
        self.model.add(layers.Bidirectional(
            layers.GRU(units=100, dropout=0.2, recurrent_dropout=0, activation="tanh",
                       recurrent_activation="sigmoid", unroll=False, use_bias=True,
                       reset_after=True)
        ))
        self.model.add(layers.Dense(100, activation="relu")),
        self.model.add(layers.Dense(1, activation="sigmoid"))

        # Compile model
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(),  # Default learning rate is 0.001
            metrics=["accuracy"])

        # Print model summary
        print("Model summary")
        print(self.model.summary())

    def train(self, x, y, batch_size: int = 128, epochs: int = 10):
        """
        Train the model.

        :param x: input data (tweets)
        :type x: np.ndarray

        :param y:  labels
        :type y: np.ndarray

        :param batch_size: batch size for training
        :type batch_size: int. Default is 128

        :param epochs: epochs for training
        :type epochs: int. Default is 10
        """

        # Split data
        X_train, X_val, y_train, y_val = self.split_data(x, y)

        # Convert text of train and val set to sequence
        X_train_pad = self.padding(X_train)
        X_val_pad = self.padding(X_val)

        # Generate embedding matrix
        embedding_matrix = self.generate_embedding_matrix()

        # Build model
        self.build_model(embedding_matrix)

        # Train model
        print("Fitting model")
        self.model.fit(
            X_train_pad,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_pad, y_val))

        # Save the model
        print("Saving model")
        self.model.save(f"{self.weight_path}/model", save_format='h5')

        # Save the weights
        print("Saving weights")
        self.model.save_weights(f"{self.weight_path}/model_weights.h5")

    def predict(self, x):
        """
        Generate predictions.

        :param x: input data (tweets)
        :type x: np.ndarray
        """

        # Load model if `is_weight` is True
        if self.is_weight:
            self.model = keras.saving.load_model(f"{self.weight_path}/model")

        # Convert input data to sequence
        X_pad = self.padding(x)
        logits = self.model.predict(X_pad).squeeze()
        predictions = np.where(logits >= 0.5, 1, -1)

        # Save predictions
        self.submit(predictions)
