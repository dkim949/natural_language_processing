from sklearn.naive_bayes import MultinomialNB


def train_model(X_train_tfidf, y_train):
    """
    Trains a Naive Bayes model using the provided TF-IDF vectorized data.

    Args:
        X_train_tfidf: TF-IDF vectorized training data.
        y_train: Training labels.

    Returns:
        model: Trained Naive Bayes model.
    """
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model
