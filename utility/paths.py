class DataPath:
    TEST = "data/test_data.txt"

    TRAIN_NEG_FULL = "data/train_neg_full.txt"
    TRAIN_POS_FULL = "data/train_pos_full.txt"
    TRAIN_NEG = "data/train_neg.txt"
    TRAIN_POS = "data/train_pos.txt"

    # GRU
    GRU_TRAIN = "data/preprocessed/gru/train.csv"
    GRU_TEST = "data/preprocessed/gru/test.csv"
    GRU_WEIGHT = "weights/gru"
    GRU_SUBMISSION = "submissions/gru"

    # BERT
    BERT_TRAIN = "data/preprocessed/bert/train.csv"
    BERT_TEST = "data/preprocessed/bert/test.csv"
    BERT_WEIGHT = "weights/bert"
    BERT_LARGE_WEIGHT = "weights/bert-large"
    BERT_SUBMISSION = "submissions/bert"

    # RoBERTa
    ROBERTA_WEIGHT = "weights/roberta"
    ROBERTA_SUBMISSION = "submissions/roberta"

    # For other ML models
    ML_TRAIN = "data/preprocessed/ml/train.csv"
    ML_TRAIN_CROPPED = "data/preprocessed/ml/train_cropped.csv"
    ML_TEST = "data/preprocessed/ml/test.csv"
    ML_SUBMISSION = "submissions/ml"


class UtilityPath:
    SLANG = "utility/files/slang.csv"
    GLOVES = "data/glove.twitter.27B.100d.txt"
