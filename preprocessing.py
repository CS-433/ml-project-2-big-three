import pandas as pd
import numpy as np
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from importlib_resources import files
from symspellpy import SymSpell
import re

from utility.decorators import print_func_name
from utility.paths import UtilityPath, DataPath

# Setup tqdm verbose
tqdm.pandas()

# Setup nltk weights
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class PreprocessingUtils:
    def __init__(self):
        self._symspell = None

    def _get_symspell(self):
        """
        Instantiates a `SymSpell` object.

        :return: instantiated object
        :rtype: SymSpell
        """
        # If it's not already instantiated
        if self._symspell is None:
            # Instantiating `SymSpell`
            self._symspell = SymSpell()

            # Getting dictionary for single words
            dictionary_path = files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
            self._symspell.load_dictionary(
                dictionary_path, term_index=0, count_index=1
            )

            # Getting dictionary for bigram (two words)
            bigram_path = files("symspellpy").joinpath("frequency_bigramdictionary_en_243_342.txt")
            self._symspell.load_bigram_dictionary(
                bigram_path, term_index=0, count_index=2
            )

        return self._symspell

    def word_segmentation(self, text):
        """
        Tries to put spaces between words in a text (used for hashtags).
        (e.g.: helloguys --> hello guys)

        :param text: Text to be converted (typically a hashtag)
        :type text: str
        :return: Processed text
        :rtype: str
        """
        # `max_edit_distance = 0` avoids that `SymSpell` corrects spelling.
        result = self._get_symspell().word_segmentation(
            text, max_edit_distance=0
        )
        return result.segmented_string

    def correct_spelling(self, text):
        """
        Corrects the spelling of a word (e.g.: helo -> hello)

        :param text: Text to be converted
        :type text: str
        :return: Processed text
        :rtype: str
        """
        # `max_edit_distance = 2` tells `SymSpell` to check at a maximum distance
        # of 2 in the vocabulary. Only words with at most 2 letters wrong will be corrected.
        result = self._get_symspell().lookup_compound(
            text, max_edit_distance=2
        )

        return result[0].term

    def _get_wordnet_tag(self, nltk_tag):
        """
        Returns the type of word according to the nltk pos tag.

        :param nltk_tag: nltk pos tag
        :type nltk_tag: list(tuple(str, str))
        :return: type of word
        :rtype: str
        """
        if nltk_tag.startswith("V"):
            return wordnet.VERB
        elif nltk_tag.startswith("N"):
            return wordnet.NOUN
        elif nltk_tag.startswith("J"):
            return wordnet.ADJ
        elif nltk_tag.startswith("R"):
            return wordnet.ADV
        else:
            # This is the default in WordNetLemmatizer when no pos tag is passed
            return wordnet.NOUN

    def lemmatize(self, text):
        """
        Performs lemmatization using nltk pos tag and `WordNetLemmatizer`.

        :param text: Text to be processed
        :type text: str
        :return: processed text
        :rtype: str
        """
        nltk_tagged = nltk.pos_tag(text.split())
        lemmatizer = WordNetLemmatizer()

        return " ".join(
            [
                lemmatizer.lemmatize(w, self._get_wordnet_tag(nltk_tag))
                for w, nltk_tag in nltk_tagged
            ]
        )


class Preprocessing:
    def __init__(self, path_ls: list, is_test: bool = False):
        if len(path_ls) > 2 or len(path_ls) < 1:
            raise ValueError("Length of path should be 1 or 2.")

        self._is_test = is_test
        self.df = self._load_data(path_ls)
        self._prep_utils = PreprocessingUtils()

    def _load_data(self, path_ls):
        if len(path_ls) == 1 and self._is_test:
            return self._load_test_data(path_ls[0])
        else:
            return self._load_train_data(path_ls)

    def _load_train_data(self, path_ls):
        if len(path_ls) == 1 and path_ls[0].find("csv") != -1:
            return pd.read_csv(path_ls[0])

        is_neg = -1
        dfs = []

        for path in path_ls:
            if "neg" not in path:
                is_neg = 1
            with open(path) as f:
                content = f.read().splitlines()

            _df = pd.DataFrame({"text": content, "label": np.ones(len(content)) * is_neg})
            dfs.append(_df)

        df = pd.concat(dfs, ignore_index=True)
        df["text"] = df["text"].str.lower()
        df["label"] = df["label"].astype("int64")
        return df

    def _load_test_data(self, path):
        with open(path) as f:
            content = f.read().splitlines()

        ids = [line.split(",")[0] for line in content]
        texts = [",".join(line.split(",")[1:]) for line in content]

        df = pd.DataFrame({"ids": ids, "text": texts})
        df["text"] = df["text"].str.lower()
        return df

    @print_func_name
    def __get__(self) -> pd.DataFrame:
        return self.df

    @print_func_name
    def __len__(self) -> int:
        return len(self.df)

    @print_func_name
    def shape(self) -> tuple:
        return self.df.shape

    @print_func_name
    def create_raw(self):
        self.df["raw"] = self.df["text"]

    @print_func_name
    def strip(self):
        self.df["text"] = self.df["text"].str.strip()

    @print_func_name
    def remove_tag(self):
        self.df["text"] = self.df["text"].str.replace("<[\w]*>", "", regex=True)

    @print_func_name
    def remove_space_before_symbol(self):
        def _find_pattern(text):
            pattern = r'\s+([!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|])'
            return re.sub(pattern, r'\1', text)

        self.df["text"] = self.df["text"].progress_apply(_find_pattern)

    @print_func_name
    def remove_extra_space(self):
        self.df["text"] = self.df["text"].str.replace("\s{2,}", " ", regex=True)

    @print_func_name
    def remove_ellipsis(self):
        self.df["text"] = self.df["text"].str.replace(r'\.{2}$', '', regex=True)

    @print_func_name
    def remove_hashtag(self):
        self.df["text"] = self.df["text"].str.replace("#", " ")
        # self.df["text"] = self.df["text"].progress_apply(self._prep_utils._word_segmentation)

    @print_func_name
    def remove_space_after_quote(self):
        def _find_pattern(text):
            pattern = r'(("[^"]*")|(\'[^\']*\'\s))'
            return re.sub(pattern, lambda match: match.group(1).replace(' ', ''), text)

        self.df["text"] = self.df["text"].progress_apply(_find_pattern)

    @print_func_name
    def reconstruct_emoji(self, is_bert: bool = True):
        def _find_symbol(text):
            pattern = r"\s*([()])\s*"
            return re.sub(pattern, r" :\1" if text.count("(") != text.count(")") else r"\1 ", text)

        def _find_reconstruct_symbol(text) -> str:
            # Define a regular expression pattern to match emoticons
            pattern = r'(:\(|:\))'

            if is_bert:
                # Replace :) with "smile" and :( with "sad face"
                result = re.sub(r':\)', 'smile ', text)
                result = re.sub(r':\(', 'sad face ', result)
            else:
                # Use re.sub() to replace emoticons with emoticon + space
                result = re.sub(pattern, r'\1 ', text)

            return result

        self.df["text"] = self.df["text"].progress_apply(_find_symbol)
        self.df["text"] = self.df["text"].progress_apply(_find_reconstruct_symbol)

    @print_func_name
    def drop_duplicates(self):
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        # self.df = self.df.dropna()

    @print_func_name
    def lemmatize(self):
        self.df["text"] = self.df["text"].progress_apply(self._prep_utils.lemmatize)

    @print_func_name
    def correct_spelling(self):
        self.df["text"] = self.df["text"].progress_apply(self._prep_utils.correct_spelling)

    @print_func_name
    def remove_stopwords(self):
        _stopwords = set(stopwords.words("english"))

        # Removing stopwords for each tweet
        self.df["text"] = self.df["text"].progress_apply(
            lambda text: " ".join(
                [word for word in str(text).split() if word not in _stopwords]
            )
        )

    @print_func_name
    def slang_to_word(self):
        # https://github.com/Zenexer/internet-reference/blob/main/Internet%20Slang%20and%20Emoticons.md
        slang_doc = pd.read_csv(UtilityPath.SLANG).set_index('slang')['text'].to_dict()

        def _find_slang(text: str) -> str:
            new_text = []
            _default_value = "<this-is-default-value>"

            for word in text.split():
                _value = slang_doc.get(word, _default_value)
                if _value != _default_value:
                    new_text.append(_value)
                else:
                    new_text.append(word)

            return " ".join(new_text)

        self.df["text"] = self.df["text"].progress_apply(_find_slang)

    @print_func_name
    def fillna(self):
        self.df["text"] = self.df["text"].filna("<empty-text>")
