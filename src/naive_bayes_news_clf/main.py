from data_processing import load_data, vectorize_data
from model_training import train_model
from model_evaluation import evaluate_model
from roc_auc_analysis import roc_auc_analysis


def main():
    # Load and preprocess data
    newsgroups_train, newsgroups_test = load_data()
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_data(
        newsgroups_train.data, newsgroups_test.data
    )

    # Train model
    model = train_model(X_train_tfidf, newsgroups_train.target)

    # Evaluate model
    evaluate_model(
        model, X_test_tfidf, newsgroups_test.target, newsgroups_train.target_names
    )


if __name__ == "__main__":
    main()
