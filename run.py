import argparse
from argparse import RawTextHelpFormatter

import pandas as pd

from utility.paths import DataPath
from preprocessing import Preprocessing
from models.base import Model
from models.bert import BERT

from tqdm.auto import tqdm


def run_preprocessing(model: Model):
    """
    Run preprocessing steps for the model.

    :param model: The model to extract preprocessing steps on.
    :type model: Model
    """
    train_prep = Preprocessing([DataPath.TRAIN_NEG_FULL, DataPath.TRAIN_POS_FULL])
    test_prep = Preprocessing([DataPath.TEST], is_test=True)

    # Preprocessing steps
    for step in tqdm(model.preprocessing(), desc="Preprocessing train data"):
        getattr(train_prep, step)()

    for step in tqdm(model.preprocessing(is_train=False), desc="Preprocessing test data"):
        getattr(test_prep, step)()

    # Retrieve preprocessed dataframe
    train_data = train_prep.__get__()
    test_data = test_prep.__get__()

    # Sample and save train data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data.to_csv(DataPath.BERT_TRAIN, index=False)

    # Save test data
    test_data.to_csv(DataPath.BERT_TEST, index=False)


def run(args: argparse.Namespace):
    print("Initializing BERT model...")
    if args.weight:
        print("Loading pre-trained weights...")

    bert = BERT(
        weight_path=DataPath.BERT_WEIGHT,
        submission_path=DataPath.BERT_SUBMISSION,
        is_weight=args.weight
    )

    if args.preprocess:
        print("Running preprocessing steps...")
        run_preprocessing(bert)

    # Train from scratch without weights
    if not args.weight:
        # Load train df for training/validating
        print("Loading training data...")
        train_df = pd.read_csv(DataPath.BERT_TRAIN)
        train_df = train_df.dropna().reset_index(drop=True)

        batch_size = 32
        epochs = 3

        # Split train df into X and y
        X, y = train_df['text'].values, train_df['label'].values

        print("Training model from scratch...")
        bert.train(X, y, batch_size, epochs)

    # Load test df for predicting
    print("Loading test data...")
    test_df = pd.read_csv(DataPath.BERT_TEST)

    # Split text
    X_test = test_df['text']
    bert.predict(X_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script runs the BERT pre-trained with the tweet classification dataset",
        formatter_class=RawTextHelpFormatter
    )

    # Add arguments
    parser.add_argument(
        "-p",
        "--preprocess",
        action="store_true",
        help="Run preprocessing steps"
    )

    parser.add_argument(
        "-w",
        "--weight",
        action="store_true",
        help="Load the pre-trained weights"
    )

    args = parser.parse_args()

    run(args)
