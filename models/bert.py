from models.base import Model

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputFeatures, AdamWeightDecay, WarmUp
import tensorflow as tf
from keras import optimizers, losses, metrics, callbacks

import numpy as np
from tqdm.auto import tqdm


class BERT(Model):
    def __init__(self,
                 weight_path: str = "",
                 submission_path: str = "",
                 is_weight: bool = False,
                 seed: int = 42,
                 max_length: int = 128):
        super().__init__(weight_path, submission_path, is_weight, seed)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        # Load weights
        if self.is_weight:
            self.model = TFBertForSequenceClassification.from_pretrained(self.weight_path)
        else:
            self.model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

    def preprocessing(self, is_train: bool = True):
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

    def predict(self, x: np.ndarray):
        predictions = []

        for i, tweet in enumerate(tqdm(x, desc="Generating predictions")):
            feature = self.tokenizer.encode_plus(text=tweet, return_tensors='tf')
            output = self.model(feature)[0].numpy().squeeze().argmax()
            predictions.append(output)

        self.submit(predictions)

    def create_tf_dataset(self, x, y):
        features = []

        for text, label in tqdm(zip(x, y), desc="Tokenizing data", total=len(x)):
            input_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_token_type_ids=False,
                truncation=True
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

        def _generator():
            for feature in tqdm(features, desc="Generating features"):
                yield (
                    {
                        "input_ids": feature.input_ids,
                        "attention_mask": feature.attention_mask,
                    },
                    feature.label,
                )

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
        X_train, X_val, y_train, y_val = self.split_data(x, y, test_size=0.1)

        train_data = self.create_tf_dataset(X_train, y_train).shuffle(self.max_length // 2,
                                                                      reshuffle_each_iteration=True).batch(batch_size)
        val_data = self.create_tf_dataset(X_val, y_val).batch(batch_size)

        steps_per_epoch = len(X_train) // batch_size
        num_train_steps = steps_per_epoch * epochs

        print(f"Training steps: {num_train_steps}")

        lr = 2e-5
        opt_epsilon = 1.5e-8

        decay_schedule = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr,
            decay_steps=num_train_steps,
            end_learning_rate=0)

        warmup_schedule = WarmUp(
            initial_learning_rate=lr,
            decay_schedule_fn=decay_schedule,
            warmup_steps=(num_train_steps * 0.1))

        optimizer = AdamWeightDecay(learning_rate=warmup_schedule,
                                    epsilon=opt_epsilon,
                                    clipnorm=1.0)

        loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = metrics.SparseCategoricalAccuracy("accuracy")
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=self.weight_path + "/epoch-{epoch:02d}",
            save_weights_only=True,  # Set to False if you want to save the entire model
            save_best_only=False,
            save_freq='epoch')

        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric], callbacks=[checkpoint_callback])

        print("Model summary")
        print(self.model.summary())

        print("Fitting model")
        self.model.fit(train_data, epochs=epochs, validation_data=val_data)

        print("Saving weights")
        self.model.save_pretrained(self.weight_path)
