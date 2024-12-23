import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import asent
import pandas as pd


class FeatureExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("asent_en_v1")

        self.wordDict = {"NOUN": 0, "PROPN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
        self.uniqueTokens = set()
        self.sentimentalScore = None
        self.tokens = 0

    def process_text(self, text):
        doc = self.nlp(text)
        tokens = [
            token.text.lower() for token in doc if token.pos_ in self.wordDict.keys()
        ]
        self.tokens = len(tokens)

        tokens = [token for token in tokens if len(token) > 2]

        for token in doc:
            if token.pos_ in self.wordDict.keys():
                self.wordDict[token.pos_] += 1

        self.uniqueTokens.update(tokens)

        self.sentimentalScore = doc._.polarity

    def process_data(self):
        total_w = sum(
            self.wordDict.get(tag, 0) for tag in ["NOUN", "PROPN", "VERB", "ADV", "ADJ"]
        )

        # Return list of features in the same order as defined in the main function
        features = [
            (
                round(self.sentimentalScore.negative, 3)
                if self.sentimentalScore is not None
                else 0
            ),
            (
                round(self.sentimentalScore.neutral, 3)
                if self.sentimentalScore is not None
                else 0
            ),
            (
                round(self.sentimentalScore.positive, 3)
                if self.sentimentalScore is not None
                else 0
            ),
            (
                round(self.sentimentalScore.compound, 3)
                if self.sentimentalScore is not None
                else 0
            ),
            (
                self.sentimentalScore.n_sentences
                if self.sentimentalScore is not None
                else 0
            ),
            self.tokens,
            round(len(self.uniqueTokens) / self.tokens,
                  3) if self.tokens > 0 else 0,
            round(self.wordDict["NOUN"] / total_w, 3) if total_w > 0 else 0,
            round(self.wordDict["PROPN"] / total_w, 3) if total_w > 0 else 0,
            round(self.wordDict["VERB"] / total_w, 3) if total_w > 0 else 0,
            round(self.wordDict["ADV"] / total_w, 3) if total_w > 0 else 0,
            round(self.wordDict["ADJ"] / total_w, 3) if total_w > 0 else 0,
            (
                self.sentimentalScore.compound + self.sentimentalScore.neutral
                if self.sentimentalScore is not None
                and self.sentimentalScore.compound > 0
                else 0
            ),
            (
                self.tokens / self.sentimentalScore.n_sentences
                if self.sentimentalScore is not None
                and self.sentimentalScore.n_sentences > 0
                else 0
            ),
        ]

        return features
