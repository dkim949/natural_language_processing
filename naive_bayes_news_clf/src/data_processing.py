from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    """
    Loads the 20 Newsgroups dataset.
    Returns:
        newsgroups_train: The training data subset.
        newsgroups_test: The testing data subset.
    """
    newsgroups_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
    newsgroups_test = fetch_20newsgroups(subset="test", shuffle=True, random_state=42)
    return newsgroups_train, newsgroups_test


def vectorize_data(train_data, test_data, max_features=1000):
    """
    Vectorizes the training and testing data using TF-IDF.

    Args:
        train_data: List of training documents.
        test_data: List of testing documents.
        max_features: The maximum number of features to consider in TF-IDF.

    Returns:
        X_train_tfidf: TF-IDF vectorized training data.
        X_test_tfidf: TF-IDF vectorized testing data.
        tfidf_vectorizer: The fitted TF-IDF vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data)
    X_test_tfidf = tfidf_vectorizer.transform(test_data)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer
