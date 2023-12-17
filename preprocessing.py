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

EMOTICONS_GLOVE = {
    '<smile>': [':-]', '0:3', '8)', ":'-)", '>^_^<', '(^_^)', "(';')", ':*',
                '(^^)/', ':)', ':>', '(*_*)', '(^^)v', '=3', ':}', ';^)', ':->', '^_^;',
                '=)', '(^o^)', '*)', '(^.^)', '^_^', '\\o/', '^5', '(__)', '(#^.^#)', '0:)',
                '(^^)', ';]', ':-*', ':^)', ':3', '(+_+)', ';)', ":')", '(:', ':-3', ':-}',
                ';-)', ':-)', ':]', '*-)', 'o/\\o', '=]', '(^_-)', '8-)', ':o)', ':c)',
                '(^_^)/', '(o.o)', ':o', '>:)', '8-0', ':-0', ';3', '>:3', '3:)', ':-o',
                '}:)', 'o_0', '^^;', 'xx', 'xxx', '^o^', ':d', ' c:'],
    '<lolface>': [':-p', ':p', ':b', ':-b', 'x-p', '=p'],
    '<heart>': ['<3'],
    '<neutralface>': ['=\\', '>:/', '(..)', '(._.)', ':-/', ':|', '>.<', ':-.',
                      "('_')", '=/', ':/', ':#', '(-_-)', 'o-o', 'o_o', ':$', '>:\\', ':@', ':-|',
                      '><>', '(-.-)', ':\\', '<+', ':-@'],
    '<sadface>': [';(', '(~_~)', ':c', ':[', ':-&', ':(', '>:[', ':&', ':-c',
                  ';n;', ":'(", ';;', ':-[', ';-;', '%)', ':<', '<\\3', ':{', ';_;', '=(',
                  'v.v', 'm(__)m', '</3', ":'-(", ':-<']
}


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

    @staticmethod
    def _get_wordnet_tag(nltk_tag):
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
        self._path_ls = path_ls
        self._prep_utils = PreprocessingUtils()

        self.df = self._load_data()

    def _load_data(self):
        if len(self._path_ls) == 1 and self._is_test:
            return self._load_test_data()
        else:
            return self._load_train_data()

    def _load_train_data(self):
        if len(self._path_ls) == 1 and self._path_ls[0].find("csv") != -1:
            return pd.read_csv(self._path_ls[0])

        is_neg = -1
        dfs = []

        for path in self._path_ls:
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

    def _load_test_data(self):
        with open(self._path_ls[0]) as f:
            content = f.read().splitlines()

        ids = [line.split(",")[0] for line in content]
        texts = [",".join(line.split(",")[1:]) for line in content]

        df = pd.DataFrame({"ids": ids, "text": texts})
        df["text"] = df["text"].str.lower()
        return df

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
        self.strip()

    @print_func_name
    def remove_space_before_symbol(self):
        emo_list = [el for value in list(EMOTICONS_GLOVE.values()) for el in value]
        emo_with_spaces_pattern = re.compile('|'.join(re.escape(' '.join(emo)) for emo in emo_list))
        all_non_alpha_emo_pattern = re.compile(
            '|'.join(re.escape(emo) for emo in emo_list if not any(char.isalpha() or char.isdigit() for char in emo)))

        # Define a function to handle replacement
        def _replace_func(match):
            text = match.group()
            if emo_with_spaces_pattern.match(text):
                return text.replace(" ", "")
            return f' {text} '

        # Applying the transformations
        self.df["text"] = self.df["text"].progress_apply(lambda x: re.sub(all_non_alpha_emo_pattern, _replace_func, x))

    @print_func_name
    def remove_extra_space(self):
        self.df["text"] = self.df["text"].progress_apply(lambda text: " ".join(text.split()))
        self.df.reset_index(inplace=True, drop=True)

    @print_func_name
    def remove_ellipsis(self):
        self.df["text"] = self.df["text"].str.replace(r'\.{3}$', '', regex=True)

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
    def reconstruct_emoji(self):
        print("inside")

        def _find_unmatched_parentheses(text):
            open_stack = []  # Stack to keep track of indices of '('
            unmatched_indices = []  # List to store indices of unmatched parentheses

            for i, char in enumerate(text):
                if char == '(':
                    open_stack.append(i)  # Push the index of '(' onto the stack
                elif char == ')':
                    if open_stack:
                        open_stack.pop()  # Pop the last '(' as it's a matched pair
                    else:
                        unmatched_indices.append(i)  # Unmatched ')'

            # Add remaining indices from the stack to unmatched_indices
            unmatched_indices.extend(open_stack)

            return sorted(unmatched_indices)

        def _add_colon(text) -> str:
            unmatched_indices = _find_unmatched_parentheses(text)
            if len(unmatched_indices) == 0:
                return text

            char_t = list(text)

            for i, index in enumerate(unmatched_indices):
                char_t.insert(index + i, ':')

            return "".join(char_t)

        self.df["text"] = self.df["text"].progress_apply(_add_colon)

    @print_func_name
    def drop_duplicates(self):
        self.df = self.df.drop_duplicates(subset=['text'])
        self.df = self.df.dropna().reset_index(drop=True)

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
