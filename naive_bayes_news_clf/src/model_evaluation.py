from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test_tfidf, y_test, target_names):
    """
    Evaluates the trained model using the test data.

    Args:
        model: Trained Naive Bayes model.
        X_test_tfidf: TF-IDF vectorized test data.
        y_test: True labels for the test data.
        target_names: Names of the target classes.

    Prints:
        Accuracy of the model.
        Classification report showing precision, recall, F1-score for each class.
        Confusion matrix as a heatmap.
    """
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    class_report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification Report:\n")
    print(class_report)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
