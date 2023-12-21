from models.bert import BERT

from transformers import AutoTokenizer, TFRobertaForSequenceClassification


class RoBERTa(BERT):
    """
    A RoBERTa-based model for tweet classification task. Strictly inherited from BERT.
    The model is pretrained on roberta-base model, provided by
    [HuggingFace](https://huggingface.co/transformers/).
    """

    def __init__(self,
                 weight_path: str = "",
                 submission_path: str = "",
                 is_weight: bool = False,
                 seed: int = 42,
                 max_length: int = 128,
                 pretrained_model: str = "roberta-base"):
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
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = max_length

        # Load models
        if self.is_weight:
            self.model = TFRobertaForSequenceClassification.from_pretrained(self.weight_path)
        else:
            self.model = TFRobertaForSequenceClassification.from_pretrained(pretrained_model)
