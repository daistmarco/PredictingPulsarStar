![pulsar_star_wp](https://github.com/daistmarco/PredictingPulsarStar/assets/93286641/acc4eee3-d1dc-4c31-a5a4-0df5a0afec02)

# Predicting Pulsar Stars

This project aims to predict whether a candidate in the dataset is a pulsar using machine learning techniques. In practice, the detection of pulsar signals is challenging due to the presence of radio frequency interference (RFI) and noise. Almost all detections are caused by RFI and noise, making it difficult to find legitimate signals. The goal is to identify these rare types of neutron stars that emit detectable radio emissions from Earth. By accurately predicting pulsar stars, prioritization can be given to further research and analysis.

## Team Members

- Joseph Wiesemeier: [GitHub](https://github.com/Dabbeljuh)
- Marco Fischer: [GitHub](https://github.com/daistmarco)
- Kathrin Muller: [GitHub](https://github.com/KathrinMM)

## Jupyter Notebooks

This project consists of three Jupyter Notebooks that serve different purposes:

1. **PulsarStarClassification.ipynb**: This notebook focuses on Exploratory Data Analysis (EDA) and training various models for pulsar star classification. It includes data preprocessing, feature engineering, model training, and evaluation. The notebook provides insights into the data and compares the performance of different machine learning algorithms.

2. **Predictions.ipynb**: In this notebook, you will find a pre-trained model that has been selected as the best model for pulsar star classification. It can be used to make predictions on new data without the need for retraining. The notebook demonstrates how to load the trained model and perform predictions on unseen data.

3. **ANN_comparison.ipynb**: This notebook compares two Artificial Neural Network (ANN) models for pulsar star classification. It includes a baseline ANN model and an optimized version using Principal Component Analysis (PCA) and polynomial features. The notebook presents the comparison of these models, including their performance metrics and insights gained from the analysis.

## Installation and Setup

To set up the project locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/daistmarco/PredictingPulsarStar.git
```
2. Navigate to the project directory:
```
cd your-repository
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Download the modified dataset and place it in the project directory. The original dataset can be acquired from the link [Predicting a Pulsar Star](https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate).

5. Ensure that the following files are located in the "models" folder:
- `cleaner.pkl`: The saved cleaner object used for data preprocessing.
- `fitted_ANN_preprocessing_pipeline.pkl`: The saved preprocessing pipeline used for the ANN model.
- `ann.h5`: The trained Artificial Neural Network model.
- `random_forest.pkl`: The trained Random Forest Classifier model.
- `svm.pkl`: The trained Support Vector Machine model.

**Note:** If any of the above files are missing, the corresponding functionality may not work as expected.

Once the setup is complete, you can use the provided functions, such as `ann_prediction(csv_file)`, to make predictions on new data using the pre-trained models.


## Dataset

The project utilizes a modified version of the HTRU2 dataset, originally collected by Pavan Raj and available on Kaggle. Pulsars are a rare type of neutron star that produce radio emissions detectable on Earth. The dataset consists of 8 continuous variables and a single class variable.

The first four variables are simple statistics obtained from the integrated pulse profile (folded profile), representing a longitude-resolved version of the signal that has been averaged in both time and frequency. The remaining four variables are obtained from the DM-SNR curve.

For the purpose of creating a challenge, certain modifications have been made to the dataset. The modified version used in this project can be obtained from the following link: [Predicting Pulsar Star (Intermediate)](https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate).

## Attribute Information

The dataset contains the following attributes:

1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class (target variable)

The dataset contains a total of 17,898 examples, with 1,639 positive examples (pulsar stars) and 16,259 negative examples.

The credit for the dataset goes to Pavan Raj. The dataset was modified for the purpose of creating this challenge.

## EDA/Cleaning

The initial steps involved understanding the data and addressing missing values and outliers. Three columns had missing values, which were imputed using the k-nearest neighbor imputer. Exploratory Data Analysis (EDA) techniques, such as box plots and violin plots, were used to identify and analyze outliers. It was observed that the outliers belonged to the pulsar star data, and no further cleaning was performed.

## Model Choices

Three different models were chosen for the binary classification task:

1. Random Forest Classifier
2. Support Vector Machine
3. Artificial Neural Network (ANN)

Each team member trained a different model, and the performance of all models was compared to select the best model for this dataset.

## Results

The performance of the models was evaluated using the F1 score, which is the average of precision and recall. The Artificial Neural Network model with polynomial features and Principal Component Analysis achieved the highest F1 score and was selected as the best model for predicting pulsar stars.

![df_results_sc](https://github.com/daistmarco/PredictingPulsarStar/assets/114780077/f9e96535-2629-4da4-92fa-37855d96e05b)

## Prediction Function

As a result of the project, a function called ann_prediction was developed. This function allows making predictions on data from a CSV file using the pre-trained Artificial Neural Network (ANN) model. The function loads the model and performs the necessary data preprocessing before making predictions. Please refer to the function documentation for usage instructions.

Please note that the pre-trained ANN model, the cleaner object, and the preprocessing pipeline used by the ann_prediction function should be available in the models folder.

## Dataset Credit

This project uses a modified version of the HTRU2 dataset from Kaggle. The original HTRU2 dataset was sourced from the University of Manchester [School of Physics and Astronomy](https://www.physics.manchester.ac.uk/), and can be found at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HTRU2).

We would like to acknowledge and thank Dr. Robert Lyon from the University of Manchester for making this dataset available for use.

