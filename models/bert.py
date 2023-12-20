from models.base import Model

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputFeatures, AdamWeightDecay, WarmUp
import tensorflow as tf
from keras import optimizers, losses, metrics, callbacks

import numpy as np
from tqdm.auto import tqdm


class BERT(Model):
    """
    A BERT-based model for tweet classification task. The model is pretrained on BERT-base-uncased model, provided by
    [HuggingFace](https://huggingface.co/transformers/).
    """

    def __init__(self,
                 weight_path: str = "",
                 submission_path: str = "",
                 is_weight: bool = False,
                 seed: int = 42,
                 max_length: int = 128,
                 pretrained_mode: str = "bert-base-uncased"):
        """
        Initialize the BERT model with specified parameters.

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
        """

        super().__init__(weight_path, submission_path, is_weight, seed)

        # Set pretrained tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_mode)
        self.max_length = max_length

        # Load models
        if self.is_weight:
            self.model = TFBertForSequenceClassification.from_pretrained(self.weight_path)
        else:
            self.model = TFBertForSequenceClassification.from_pretrained(pretrained_mode)

    def preprocessing(self, is_train: bool = True):
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
            "remove_tag",
            "remove_ellipsis",
            "reconstruct_emoji",
            "remove_extra_space",
            "remove_space_around_emoji",
            "remove_extra_space"
        ]

        if is_train:
            return ["drop_duplicates"] + steps

        return steps

    def create_tf_dataset(self, x: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
        """
        Tokenization of each document up to max_length, generating a TensorFlow dataset.
        Each element includes a dictionary with tokenized text and attention mask, and the tweet's label.

        :param x: Input data (tweets).
        :type x: np.ndarray

        :param y: Labels.
        :type y: np.ndarray

        :return: A TensorFlow dataset.
        :rtype: tf.data.Dataset
        """
        # A list of `InputFeatures`. An `InputFeatures` consists of
        # tweet's tokens (`input_ids`), tweet's attention mask (`attention_mask`), tweet's label (`label`).
        features = []

        for text, label in tqdm(zip(x, y), desc="Tokenizing data", total=len(x)):
            input_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,  # fixed length
                padding="max_length",  # pad to max_length
                # Returns a binary vector of length `max_length`,
                # where 1 indicates a valid token in the tweet and 0 represents a padding character.
                # More at https://huggingface.co/transformers/glossary.html#attention-mask.
                return_attention_mask=True,  # return attention mask
                # False since we don't compare text. More at
                # https://huggingface.co/docs/transformers/glossary#token-type-ids
                return_token_type_ids=False,
                truncation=True  # truncates if len(s) > max_length
            )

            input_ids, attention_mask = (
                input_dict['input_ids'],
                input_dict['attention_mask'])

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask,
                    label=label
                )
            )

        # A generator that yields the features
        def _generator():
            for feature in tqdm(features, desc="Generating features"):
                yield (
                    {
                        "input_ids": feature.input_ids,
                        "attention_mask": feature.attention_mask,
                    },
                    feature.label,
                )

        # Create a TensorFlow dataset
        return tf.data.Dataset.from_generator(
            _generator,
            ({
                 "input_ids": tf.int32,
                 "attention_mask": tf.int32,
             }, tf.int64),
            ({
                 "input_ids": tf.TensorShape([None]),
                 "attention_mask": tf.TensorShape([None]),
             }, tf.TensorShape([]),),
        )

    def train(self, x, y, batch_size: int, epochs: int):
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

        # Split data
        X_train, X_val, y_train, y_val = self.split_data(x, y, test_size=0.1)

        # Create TensorFlow datasets
        train_data = self.create_tf_dataset(X_train, y_train).shuffle(self.max_length // 2,
                                                                      reshuffle_each_iteration=True).batch(batch_size)
        val_data = self.create_tf_dataset(X_val, y_val).batch(batch_size)

        # Calculate training steps
        steps_per_epoch = len(X_train) // batch_size
        num_train_steps = steps_per_epoch * epochs

        print(f"Training steps: {num_train_steps}")

        # Optimizer
        lr = 2e-5
        opt_epsilon = 1.5e-8

        # Apply a polynomial decay to the learning rate
        decay_schedule = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr,
            decay_steps=num_train_steps,
            end_learning_rate=0)

        # Apply a warmup schedule on a given learning rate decay schedule
        warmup_schedule = WarmUp(
            initial_learning_rate=lr,
            decay_schedule_fn=decay_schedule,
            warmup_steps=(num_train_steps * 0.1))

        # Adam optimizer with weight decay as used in BERT paper
        # More at https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adamweightdecay
        optimizer = AdamWeightDecay(learning_rate=warmup_schedule,
                                    epsilon=opt_epsilon,
                                    clipnorm=1.0)

        # Define loss and metric
        loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = metrics.SparseCategoricalAccuracy("accuracy")

        # Add checkpoint callback for each epoch
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=self.weight_path + "/epoch-{epoch:02d}",
            save_weights_only=True,  # Set to False if you want to save the entire model
            save_best_only=False,
            save_freq='epoch')

        # Compile model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # Print model summary
        print("Model summary")
        print(self.model.summary())

        # Train model
        print("Fitting model")
        self.model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[checkpoint_callback])

        # Save the model weights
        print("Saving weights")
        self.model.save_pretrained(self.weight_path)

    def predict(self, x: np.ndarray):
        """
        Generate predictions.

        :param x: input data (tweets)
        :type x: np.ndarray
        """
        predictions = []

        for i, tweet in enumerate(tqdm(x, desc="Generating predictions")):
            # Encode the tweet for prediction
            feature = self.tokenizer.encode_plus(text=tweet, return_tensors='tf')

            # The predicted label is determined by taking the argmax (either 0 or 1) of this array.
            output = self.model(feature)[0].numpy().squeeze().argmax()
            predictions.append(output)

        # Save predictions
        print("Saving predictions")
        self.submit(predictions)

        print("Finished running!")
