import pandas as pd
from tensorflow.keras.models import load_model
import joblib


def ann_prediction(csv_file):
    """
    Make predictions on data from a CSV file using a pre-trained ANN model.

    This function loads a pre-trained ANN model and uses it to make predictions on data from a CSV file. 
    The data is preprocessed using a saved cleaner object and a saved preprocessing pipeline before being fed into the model. 
    If the input data has 9 columns, it is assumed that the last column is the target column and it is removed. 
    The predictions are returned as an array of binary values.

    Args:
        csv_file (str): The path to the CSV file containing the data to make predictions on.

    Returns:
        np.ndarray: An array of binary predictions.
    """
    new_columns = ['Profile_mean', 'Profile_std', 'Profile_kurtosis', 'Profile_skewness',
                   'DM_SNR_mean', 'DM_SNR_std', 'DM_SNR_kurtosis', 'DM_SNR_skewness', 'target']

    cleaner_loaded = joblib.load('models/cleaner.pkl')
    pipeline_loaded = joblib.load(
        'models/fitted_ANN_preprocessing_pipeline.pkl')

    X_test = pd.read_csv(csv_file)

    # assumes untreated dataframe with old column names and target column
    if X_test.shape[1] == 9:
        X_test.columns = new_columns
        X_test = X_test.iloc[:, :-1]

    X_test_clean = pd.DataFrame(
        cleaner_loaded.transform(X_test), columns=X_test.columns)

    X_test_preprocessed = pipeline_loaded.transform(X_test_clean)

    loaded_model = load_model('models/ann.h5', compile=False)

    predictions = (loaded_model.predict(
        X_test_preprocessed) > 0.5).astype('int8')

    return predictions
