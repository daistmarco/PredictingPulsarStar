import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import joblib


def make_prediction(csv_file):
    """
    Make predictions on data from a CSV file using a pre-trained ANN model.

    This function loads a pre-trained ANN model and uses it to make predictions on data from a CSV file. The data is preprocessed using a saved cleaner object and a saved preprocessing pipeline before being fed into the model. The predictions are returned as an array of binary values.

    Args:
        csv_file (str): The path to the CSV file containing the data to make predictions on.

    Returns:
        np.ndarray: An array of binary predictions.
    """
    cleaner_loaded = joblib.load('models/cleaner.pkl')
    pipeline_loaded = joblib.load(
        'models/fitted_ANN_preprocessing_pipeline.pkl')

    X_test = pd.read_csv(csv_file)
    X_test_clean = pd.DataFrame(
        cleaner_loaded.transform(X_test), columns=X_test.columns)

    X_test_preprocessed = pipeline_loaded.transform(X_test_clean)

    custom_objects = {'f1_score_ann': f1_score_ann}
    with tf.keras.utils.custom_object_scope(custom_objects):
        loaded_model = load_model('models/ann.h5')

    predictions = (loaded_model.predict(
        X_test_preprocessed) > 0.5).astype('int8')

    return predictions


def f1_score_ann(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

# def clean_data(cleaner_path, df):
#     """
#     Transforms a dataframe using a saved ColumnTransformer object.

#     Args:
#         cleaner_path (str): The file path to the saved ColumnTransformer object.
#         df (pd.DataFrame): The dataframe to be transformed.

#     Returns:
#         pd.DataFrame: A new dataframe with the transformed data.

#     Example:
#         X_train_transformed = transform_data('models/cleaner.pkl', X_train)
#     """
#     # Load the fitted ColumnTransformer from the file
#     with open(cleaner_path, 'rb') as f:
#         cleaner = pickle.load(f)

#     # Define the column names
#     num_cols = ['Profile_mean', 'Profile_std', 'Profile_kurtosis', 'Profile_skewness',
#                 'DM_SNR_mean', 'DM_SNR_std', 'DM_SNR_kurtosis', 'DM_SNR_skewness']

#     # Apply the loaded ColumnTransformer to transform the data
#     df_clean = pd.DataFrame(cleaner.transform(df), columns=num_cols)

#     return df_clean


# def preprocess_data(preprocess_path, df):
#     with open(preprocess_path, 'rb') as f:
#         preprocess = pickle.load(f)

#     df_transformed = pd.DataFrame(preprocess.transform(df))

#     return df_transformed


# def make_prediction(df):
#     df_transformed = clean_data('models/cleaner.pkl', df)
#     df_preprocess = preprocess_data(
#         'models/preprocess_poly_scaler_pca.pkl', df_transformed)

#     custom_objects = {'f1_score_ann': f1_score_ann}
#     with tf.keras.utils.custom_object_scope(custom_objects):
#         loaded_model = load_model('models/ann.h5')

#     predictions = (loaded_model.predict(df_preprocess) > 0.5).astype('int8')

#     return predictions
