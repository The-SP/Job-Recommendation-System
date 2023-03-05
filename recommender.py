import numpy as np
from collections import Counter


class TfidfVectorizer:
    def __init__(self, stop_words=None):
        self.stop_words = set(stop_words) if stop_words else set()
        self.vocabulary_ = {}
        self.idf_ = []

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def fit(self, documents):
        # Get the number of documents
        N = len(documents)
        # Initialize the vocabulary
        vocabulary = {}
        # Initialize the document frequency dictionary
        df = {}
        # Loop through each document
        for document in documents:
            # Tokenize the document
            tokens = document.split()
            # Remove stop words
            tokens = [token for token in tokens if token not in self.stop_words]
            # Get the word count in the document
            word_counts = Counter(tokens)
            # Update the vocabulary
            for word, count in word_counts.items():
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)
                    df[word] = 0
                df[word] += 1
        # Compute the idf values
        self.idf_ = [np.log(N / df[word]) for word in vocabulary]
        self.vocabulary_ = vocabulary
        return self

    def transform(self, documents):
        # Get the number of documents
        N = len(documents)
        # Get the number of words in the vocabulary
        M = len(self.vocabulary_)
        # Initialize the tf-idf matrix
        tfidf_matrix = np.zeros((N, M))
        # Loop through each document
        for i, document in enumerate(documents):
            # Tokenize the document
            tokens = document.split()
            # Remove stop words
            tokens = [token for token in tokens if token not in self.stop_words]
            # Get the word count in the document
            word_counts = Counter(tokens)
            # Loop through each word in the document
            for word, count in word_counts.items():
                j = self.vocabulary_.get(word, -1)
                if j >= 0:
                    tf = count / len(tokens)
                    tfidf_matrix[i, j] = tf * self.idf_[j]
        return tfidf_matrix


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


english_stopwords = ""
# open the file in read mode
with open("recommender/stopwords.txt", "r") as file:
    # read the contents of the file into a string variable
    english_stopwords = file.read().split()
