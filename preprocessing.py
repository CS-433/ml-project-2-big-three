import pandas as pd
import numpy as np
import spacy

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
        Corrects the spelling of words in the text. Skips single character words.

        :param text: Text to be corrected
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
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model once during initialization


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
    def word_segmentation(self):
        self.df["text"] = self.df["text"].progress_apply(self._prep_utils.word_segmentation)

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


    @print_func_name
    def correct_spacing_indexing(self):
        """
        Deletes double or more spaces and corrects indexing.

        Must be called after calling the above methods.
        Most of the above methods just delete a token. However since tokens are
        surrounded by whitespaces, they will often result in having more than one
        space between words.

        The only exception is for `remove_space_between_emoticons` method.
        Should be called before and after calling that method.
        It could exist ':  )' which that method doesn't recognize.
        """

        print("Correcting spacing...")

        # Removing double spaces
        self.df["text"] = self.df["text"].str.replace("\s{2,}", " ")

        # Stripping text
        self.df["text"] = self.df["text"].apply(lambda text: text.strip())

        # Correcting the indexing
        self.df.reset_index(inplace=True, drop=True)

    @print_func_name
    def remove_space_between_emoticons(self):
        """
        Removes spaces between emoticons (e.g.: ': )' --> ':)').
        Adds a space between a word and an emoticon (e.g.: 'hello:)' --> 'hello :)')
        """

        print("Removing space between emoticons...")

        # Getting list of all emoticons
        emo_list = [el for value in list(EMOTICONS_GLOVE.values()) for el in value]

        # Putting a space between each character in each emoticon
        emo_with_spaces = "|".join(re.escape(" ".join(emo)) for emo in emo_list)

        # Getting all emoticons that don't contain any alphanumeric character
        all_non_alpha_emo = "|".join(
            re.escape(emo)
            for emo in emo_list
            if not any(char.isalpha() or char.isdigit() for char in emo)
        )

        # Removing spaces between emoticons
        self.df["text"] = self.df["text"].str.replace(
            emo_with_spaces, lambda t: t.group().replace(" ", ""), regex=True
        )

        # Adding space between a word and an emoticon
        self.df["text"] = self.df["text"].str.replace(
            rf"({all_non_alpha_emo})", r" \1 ", regex=True
        )

    @print_func_name
    def emoticons_to_tags(self):
        # Dictionary like {tag:[list_of_emoticons]}
        union_re = {}
        for tag, emo_list in EMOTICONS_GLOVE.items():
            # Getting emoticons as they are
            re_emo = "|".join(re.escape(emo) for emo in emo_list)
            union_re[tag] = re_emo

        # Function to be called for each tweet
        def _inner(text, _union_re):
            for tag, union_re in _union_re.items():
                text = re.sub(union_re, " " + tag + " ", text)
            return text

        # Applying for each tweet
        self.df["text"] = self.df["text"].apply(lambda x: _inner(str(x), union_re))

    @print_func_name
    def hashtags_to_tags(self):
        """
        Convert hashtags. (e.g.: #hello ---> <hashtag> hello)
        """

        print("Converting hashtags to tags...")
        self.df["text"] = self.df["text"].str.replace(
            r"#(\S+)", r"<hashtag> \1"
        )

    @print_func_name
    def repeat_to_tags(self):
        """
        Convert repetitions of '!' or '?' or '.' into tags.
          (e.g.: ... ---> . <repeat>)
        """

        print("Converting repetitions of symbols to tags...")
        self.df["text"] = self.df["text"].str.replace(
            r"([!?.]){2,}", r"\1 <repeat>"
        )

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
    def numbers_to_tags(self):
        """
        Convert numbers into tags. (e.g.: 34 ---> <number>)
        Adds a space before and after the tag to ensure it is separated from other text.
        """
        # Replace numbers with <number> tag and add spaces around the tag
        self.df["text"] = self.df["text"].str.replace(
            r"([-+]?[.\d]*[\d]+[:,.\d]*)", r" <number> ", regex=True
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
                        result += s[i+1:j]
                        i = j + 1
                    else:
                        result += s[i]
                        i += 1
                else:
                    result += s[i]
                    i += 1
            return result

        self.df["text"] = self.df["text"].apply(remove_parentheses_from_sentence)
