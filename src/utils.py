from sklearn.metrics import classification_report

def get_classification_report(y_true, y_pred, target_names):
    """
    Generate a classification report.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - target_names: list of target names

    Returns:
    - report: classification report as a string
    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return report

def load_pipeline(path='models/logreg_pipeline.pkl'):
    """
    Load a pre-trained pipeline from a file.

    """
    import joblib
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline file not found at {path}")

    pipeline = joblib.load(path)
    return pipeline

def preprocess_new_data(df, columns):
    """
    Preprocess the data for the pipeline.

    Parameters:
    - data: DataFrame to preprocess

    Returns:
    - preprocessed_data: preprocessed DataFrame
    """
    # Example preprocessing steps
    # This should be customized based on the actual preprocessing needed
    data = data.dropna()  # Drop missing values
    return data