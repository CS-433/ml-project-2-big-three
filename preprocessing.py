import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from importlib.resources import files
from symspellpy import SymSpell
import re

from utility.decorators import print_func_name
from utility.paths import UtilityPath

# Setup tqdm verbose
tqdm.pandas(leave=True, position=0)

# Setup nltk weights
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Emojis taken from this link: https://en.wikipedia.org/wiki/List_of_emoticons. The tags are from GloVe embedding.
EMOJI_GLOVE = {
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
        words = text.split()
        segmented_words = []

        for word in words:
            # Check if the word contains at least 7 characters
            if len(word) >= 7:
                # Apply word segmentation
                result = self._get_symspell().word_segmentation(word, max_edit_distance=0)
                segmented_words.append(result.segmented_string)
            else:
                segmented_words.append(word)

        return ' '.join(segmented_words)

    def correct_spelling(self, text):
        """
        Corrects the spelling of a word (e.g.: helo -> hello)

        :param text: Text to be converted
        :type text: str

        :return: Processed text
        :rtype: str
        """
        words = text.split()
        corrected_words = []

        for word in words:
            # Skip correction if the word is a single character
            if len(word) <= 1:
                corrected_words.append(word)
                continue

            # Perform correction for words with more than one character
            result = self._get_symspell().lookup_compound(word, max_edit_distance=1)
            corrected_words.append(result[0].term)

        return ' '.join(corrected_words)

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
    """
    Class that performs preprocessing on a dataset.
    """

    def __init__(self, path_ls: list, is_test: bool = False):
        """
        Initialize the class.

        :param path_ls: list of paths to the dataset
        :type path_ls: list(str)

        :param is_test: whether the dataset is a test dataset or not
        :type is_test: bool. Default is False
        """
        if len(path_ls) > 2 or len(path_ls) < 1:
            raise ValueError("Length of path should be 1 or 2.")

        self._is_test = is_test
        self._path_ls = path_ls
        self._prep_utils = PreprocessingUtils()
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model once during initialization
        self.df = self._load_data()

    def _load_data(self):
        """
        Loads the data from the path.
        """
        if len(self._path_ls) == 1 and self._is_test:
            return self._load_test_data()
        else:
            return self._load_train_data()

    def _load_train_data(self):
        """
        Loads the training data from the path.
        """
        if len(self._path_ls) == 1 and self._path_ls[0].find("csv") != -1:
            return pd.read_csv(self._path_ls[0])

        # Flag to check if the dataset is negative or not
        is_neg = 0
        dfs = []

        for path in self._path_ls:
            if "neg" not in path:
                is_neg = 1
            with open(path) as f:
                content = f.read().splitlines()

            _df = pd.DataFrame({"text": content, "label": np.ones(len(content)) * is_neg})
            dfs.append(_df)

        df = pd.concat(dfs, ignore_index=True)
        df["text"] = df["text"].str.lower()  # Lower the text
        df["label"] = df["label"].astype("float64")
        return df

    def _load_test_data(self):
        """
        Loads the test data from the path.
        """
        with open(self._path_ls[0]) as f:
            content = f.read().splitlines()

        ids = [line.split(",")[0] for line in content]
        texts = [",".join(line.split(",")[1:]) for line in content]

        df = pd.DataFrame({"ids": ids, "text": texts})
        df["text"] = df["text"].str.lower()  # Lower the text
        return df

    def __get__(self) -> pd.DataFrame:
        """
        Returns the dataframe.
        """

        return self.df

    @print_func_name
    def __len__(self) -> int:
        """
        Returns the length of the dataframe.

        :return: length of the dataframe
        :rtype: int
        """

        return len(self.df)

    @print_func_name
    def shape(self) -> tuple:
        """
        Returns the shape of the dataframe.

        :return: shape of the dataframe
        :rtype: tuple
        """

        return self.df.shape

    @print_func_name
    def create_raw(self):
        """
        Creates a column called `raw` which contains the original text.
        """

        self.df["raw"] = self.df["text"]

    @print_func_name
    def strip(self):
        """
        Strips the text.
        """

        self.df["text"] = self.df["text"].str.strip()

    @print_func_name
    def remove_tag(self):
        """
        Removes tags bounded by `<` and `>` from the text.
        :return:
        """

        self.df["text"] = self.df["text"].str.replace(r"<[\w]*>", "", regex=True)

        # Strip after remove tags
        self.strip()

    @print_func_name
    def remove_space_around_emoji(self):
        """
        Removes spaces around emojis. (e.g.: " : ) " -> ":)")
        Add space between a word and an emoji. (e.g.: "hello:)world" -> "hello :) world")
        """

        emo_list = [el for value in list(EMOJI_GLOVE.values()) for el in value]

        # Match spaced-out emojis in text
        emo_with_spaces = '|'.join(re.escape(' '.join(emo)) for emo in emo_list)

        # Match emojis in text that are symbolic
        symbol_emo = '|'.join(re.escape(emo) for emo in emo_list if not any(
            char.isalpha() or char.isdigit() for char in emo))

        # Removing spaces between emojis
        self.df["text"] = self.df["text"].str.replace(emo_with_spaces, lambda t: t.group().replace(' ', ''), regex=True)

        # Adding space between a word and an emoticon
        self.df["text"] = self.df["text"].str.replace(rf'({symbol_emo})', r' \1 ', regex=True)

    @print_func_name
    def remove_extra_space(self):
        """
        Removes extra spaces from the text.
        """

        self.df["text"] = self.df["text"].str.replace(r'\s{2,}', ' ', regex=True)
        self.df["text"] = self.df["text"].progress_apply(lambda text: text.strip())
        self.df.reset_index(inplace=True, drop=True)

    @print_func_name
    def remove_ellipsis(self):
        """
        Removes ellipsis from the text.
        """

        self.df["text"] = self.df["text"].str.replace(r"\.{3}$", "", regex=True)

    @print_func_name
    def remove_ending(self):
        """
        Removes the ending similar to "... <url>" of the text.
        """

        self.df["text"] = self.df["text"].str.replace(r"\.{3} <url>$", "", regex=True)

    @print_func_name
    def remove_hashtag(self):
        """
        Removes hashtags from the text.
        :return:
        """

        self.df["text"] = self.df["text"].str.replace("#", " ")
        # self.df["text"] = self.df["text"].progress_apply(self._prep_utils._word_segmentation)

    @print_func_name
    def remove_space_after_quote(self):
        """
        Removes space after quote. (e.g.: "hello 'world' " -> "hello 'world'")
        """

        # Matching pattern
        def _find_pattern(text):
            pattern = r'(("[^"]*")|(\'[^\']*\'\s))'
            return re.sub(pattern, lambda match: match.group(1).replace(' ', ''), text)

        self.df["text"] = self.df["text"].progress_apply(_find_pattern)

    @print_func_name
    def reconstruct_emoji(self):
        """
        Reconstructs emojis. (e.g.: ")" -> ":)")
        """

        def _find_unmatched_parentheses(text: str) -> list:
            """
            Finds unmatched parentheses in a text.
            :param text: Text to be processed
            :type text: str

            :return: List of indices of unmatched parentheses
            :rtype: list
            """

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

        def _add_colon(text: str) -> str:
            """
            Adds colon to unmatched parentheses.

            :param text: Text to be processed
            :type text: str

            :return: Processed text
            :rtype: str
            """

            unmatched_indices = _find_unmatched_parentheses(text)
            if len(unmatched_indices) == 0:
                return text

            char_t = list(text)

            for i, index in enumerate(unmatched_indices):
                char_t.insert(index + i, ':')

            return "".join(char_t)

        self.df["text"] = self.df["text"].progress_apply(_add_colon)

    @print_func_name
    def reconstruct_last_emoji(self):
        """
        Reconstructs the last emoji. (e.g.: "hello )" -> "hello <smile>")
        """

        self.df["text"] = self.df["text"].str.replace(r'\)+$', ' <smile> ', regex=True)
        self.df["text"] = self.df["text"].str.replace(r'\(+$', ' <sadface> ', regex=True)

    @print_func_name
    def drop_duplicates(self):
        """
        Drops duplicates from the dataframe.
        """

        self.df = self.df.drop_duplicates(subset=['text'])
        self.df = self.df.dropna().reset_index(drop=True)

    @print_func_name
    def lemmatize(self):
        """
        Performs lemmatization on the text.
        """

        self.df["text"] = self.df["text"].progress_apply(self._prep_utils.lemmatize)

    @print_func_name
    def word_segmentation(self):
        self.df["text"] = self.df["text"].progress_apply(self._prep_utils.word_segmentation)

    @print_func_name
    def correct_spelling(self):
        """
        Corrects the spelling of the text.
        """

        self.df["text"] = self.df["text"].progress_apply(self._prep_utils.correct_spelling)

    @print_func_name
    def remove_stopwords(self):
        """
        Removes stopwords from the text.
        """

        _stopwords = set(stopwords.words("english"))

        # Removing stopwords for each tweet
        self.df["text"] = self.df["text"].progress_apply(
            lambda text: " ".join(
                [word for word in str(text).split() if word not in _stopwords]
            )
        )

    @print_func_name
    def emoji_to_tag(self):
        """
        Replaces emojis with their tags. Tags are based on GloVe embedding.
        """

        union = {tag: '|'.join(re.escape(emo) for emo in emo_list) for tag, emo_list in EMOJI_GLOVE.items()}

        # Function to be called for each tweet
        def _replace(text: str) -> str:
            """
            Replaces emojis with their tags.
            :param text: Text to be processed
            :type text: str

            :return: Processed text
            :rtype: str
            """
            for _tag, _union in union.items():
                text = re.sub(_union, f" {_tag} ", text)
            return text

        # Applying for each tweet
        self.df["text"] = self.df["text"].progress_apply(_replace)

    @print_func_name
    def num_to_tag(self):
        """
        Replaces numbers with a tag.
        """

        self.df["text"] = self.df["text"].str.replace(r'[-+]?[.\d]*[\d]+[:,.\d]*', r'<number>', regex=True)

    @print_func_name
    def hashtag_to_tag(self):
        """
        Replaces hashtags with a tag.
        """

        self.df["text"] = self.df["text"].str.replace(r'#(\S+)', r'<hashtag> \1', regex=True)

    @print_func_name
    def repeat_symbols_to_tag(self):
        """
        Replaces repeated symbols with a tag.
        """

        self.df["text"] = self.df["text"].str.replace(r'([!?.]){2,}', r'\1 <repeat>', regex=True)

    @print_func_name
    def elongate_to_tag(self):
        """
        Replaces elongated words with a tag.
        """

        self.df["text"] = self.df["text"].str.replace(r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <elong>', regex=True)

    @print_func_name
    def slang_to_word(self):
        """
        Replaces slang with the corresponding word.
        Our slang dictionary is taken from this link:
        https://github.com/Zenexer/internet-reference/blob/main/Internet%20Slang%20and%20Emoticons.md
        """

        slang_doc = pd.read_csv(UtilityPath.SLANG).set_index('slang')['text'].to_dict()

        def _find_slang(text: str) -> str:
            """
            Replaces slang with the corresponding word.
            :param text:
            :return:
            """
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
        """
        Fills NaN values with "<empty-text>".
        :return:
        """

        self.df["text"] = self.df["text"].filna("<empty-text>")

    @print_func_name
    def remove_selected_characters(self):
        """
        Remove selected characters from the text.
        """

        print("Removing selected characters...")
        self.df["text"] = self.df["text"].str.replace(
            r'[^a-zA-Z0-9.!\-()]', ' ', regex=True
        )

    @print_func_name
    def replace_entities_with_tags(self):
        # Define a function that will be applied to each text entry
        def replace_entities(text, nlp):
            doc = nlp(text)
            # Perform replacements
            for ent in reversed(doc.ents):
                if ent.label_ == "PERSON":
                    text = text[:ent.start_char] + '<firstname>' + text[ent.end_char:]
                elif ent.label_ in {"GPE", "LOC"}:
                    text = text[:ent.start_char] + '<city_or_country>' + text[ent.end_char:]
            return text

        # Apply the replace_entities function to each row in the DataFrame
        self.df['text'] = self.df['text'].apply(replace_entities, nlp=self.nlp)

    @print_func_name
    def remove_parentheses(self):
        """
        :rtype: None
        """

        def remove_parentheses_from_sentence(s):
            """
            :type s: str
            :rtype: str
            """
            result = ""
            i = 0
            while i < len(s):
                if s[i] == "(":
                    j = i
                    while j < len(s) and s[j] != ")":
                        if s[j] == "(":
                            result += s[i:j]
                            i = j
                        j += 1
                    if j < len(s) and s[j] == ")":
                        result += s[i + 1:j]
                        i = j + 1
                    else:
                        result += s[i]
                        i += 1
                else:
                    result += s[i]
                    i += 1
            return result

        self.df["text"] = self.df["text"].apply(remove_parentheses_from_sentence)
