# Natural Language Processing Projects

This repository contains various projects focused on different aspects of Natural Language Processing (NLP). Each subproject is organized in its own directory and tackles a specific NLP problem or use case.

## Repository Structure
**`naive_bayes_news_clf/`**: 
  - **Description**: A subproject focused on building a Naive Bayes classifier to categorize news articles into different topics using the 20 Newsgroups dataset.
  - **Structure**:
    - **`data/`**: Specific data used for this subproject.
    - **`notebooks/`**: Notebooks related to this particular project, including any initial analysis or prototyping.
    - **`src/`**: Contains scripts specific to the Naive Bayes classifier, including data processing, model training, and evaluation.
      - **`data_processing.py`**: Handles the loading and preprocessing of data.
      - **`model_training.py`**: Contains the training logic for the Naive Bayes model.
      - **`model_evaluation.py`**: Includes evaluation metrics and confusion matrix generation.
      - **`main.py`**: Orchestrates the data processing, model training, and evaluation steps.
      - **`run.py`**: Entry point to execute the entire Naive Bayes classification pipeline.
  - **Usage**: Navigate to the `naive_bayes_news_clf` directory and run the `run.py` script to train and evaluate the Naive Bayes model on the 20 Newsgroups dataset.
  - **Example**:
    ```bash
    cd naive_bayes_news_clf
    python run.py
    ```
