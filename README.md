# Predicting Pulsar Stars

The project predicting pulsar stars comes from a competition from the organisation Kaggle which has already taken place before starting this project. The project files and the machine learning under taken were all done using Python and presented in Notebooks. Due to many Neutron Stars there is a lot of data to be researched to find the rare type of neutron star, a pulsar which produces radio emission detecable from Earth. Testing all potential pulsars stars is time consuming, this leads to the goal of the project which is predict if a neutron star is also a pulsar star which can lead to prioritising pedicted pulsar stars over others. 


insert participants


### Installation and Setup
This project provides tools for analyzing the HTRU2 dataset using machine learning techniques. One of the main functions provided is the ann_prediction function, which allows you to make predictions on new data using a pre-trained ANN model. These are the necessary components you are need in order to set the project up localy. The data is linked within the information below which was provided from kaggle that leads you to the changed data set provided by kaggle which had been altered for this competition, where the original data set from the University of Manchester has no missing values.

The project was carried out in JupyterLab using Python Version 3.9 and 3.11 interchangeably.

Listed below are all necessary dependencies you require to replicate the project yourself:
- os
- pickle
- pandas
- numpy
- seaborn
- matplotlib
- IPython
- sklearn
- imblearn
- tensorflow
- keras
- xgboost
- joblib

To handle the data in the same state as we have you will need to clean it using the functions that are within the provided notebooks if you wish to go through the steps, otherwise the ANN prediction model has been exported after being trained and is ready to use and is found in the models folder.

### Description
The following information is provided from the challenge page:


Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter. Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted,which treat the candidate data sets as binary classification problems.

Credit goes to Pavan Raj ( https://www.kaggle.com/pavanraj159) from where the dataset has been collected. For the purpose of creating a challenge, certain modifications have been done to the dataset.
Original dataset can be acquired from the link Predicting a Pulsar Star ( https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star)
Attribute Information:

Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency . The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below:

	1.Mean of the integrated profile.
	2.Standard deviation of the integrated profile.
	3.Excess kurtosis of the integrated profile.
	4.Skewness of the integrated profile.
	5.Mean of the DM-SNR curve.
	6.Standard deviation of the DM-SNR curve.
	7.Excess kurtosis of the DM-SNR curve.
	8.Skewness of the DM-SNR curve.
	9.Class

HTRU 2 Summary

17,898 total examples.

1,639 positive examples.

16,259 negative examples.

Source:  https://archive.ics.uci.edu/ml/datasets/HTRU2

Dr Robert Lyon

University of Manchester

School of Physics and Astronomy

Alan Turing Building

Manchester M13 9PL

United Kingdom

robert.lyon@manchester.ac.uk

From <https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate> 


### Terminology of data explained
• Integrated Profile
"Integrated Profile is the signal obtained from folding/Integrating the pulsar signals w.r.t rotational period"
From <https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate/discussion/191741> 
This is a unique measurement and can be seen as an equivalent to a thumbprint.

• DM-SNR Curve
"Dispersion Measure of the Signal to Noise Ratio (column density of free electrons along the line of sight)"
From <https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate/discussion/191741> 

• Mean
The average of the numbers: sum of sample size  / sample size count which can be easily influenced by outliers. Extreme values on either end of the spectrum.

• Std. Dev
The standard deviation is a measurement of how the data is dispered in relation to the mean. Mean is the 50% benchmark while +1 std dev away is 75% and -1 is 25% benchmarks.

• Excess Kurtosis
The excess kurtosis describes the tails of the bell-shaped distribution curve which having lots of outliers will cause it to have a fat tails.

• Skewness
Skewness is a measurement used to describe the symmetry or asymmetry of the bell-curve.

![skw](https://github.com/daistmarco/PredictingPulsarStar/assets/114780077/aca3d64e-dd38-4251-9332-72d8896e3c64)

### EDA/Cleaning
First steps carried out naturally was to get acquainted with the data itself. Meaning an understanding of what and where the data came from along with finding any potential hinderance from missing values, implausible values along with an overview of how to deal with potential outliers that could strongly influence the training processes.

From the EDA we established that there were missing values within three columns; Excess kurtosis of the integrated profile, Standard deviation of the DM-SNR curve and Skewness of the DM-SNR curve. They were dealt with by implementing k-nearest neighbour imputer within the pipeline process down the line which would insert the mean value the nearest five data points of the missing value.. Removing the rows with missing values would cause to great of a loss of data from a smaller sized data set.

From the pair plot we established more visuals to get an overview on potential outliers and if they could prove to cause noise down the line. To get this understanding box plots split by the target class for a consise overview along with a violin plot of the whole column itself. The visuals strongly demonstrated that the data points that acted as outliers where from the data of pulsar stars which is our target therefore the decision was made to keep them and conduct no further cleaning on the data.


### Model Choices
Three varying models where chosen based on the aim of the project to predict a binary classification outcome;
	1. Random Forest Classifier
	2. Support Vector Machine
	3. Artificial Neural Network
All three algorithms were trained by a different team member with the goal of achieving the most successful predictions and a comparison to be made afterwards to chose the best suited model for this data set and target. 


### Results
The visuals below show the final results from all three different algorithms with the Artificial Neural Network with polynomial features along with Principal Component Analysis was found to return the best results in predicting pulsar stars. The results are based on having used the models with a test set which was not used to improve any of the models to allow us to get a more accurate estimation of how it performs and of course the final aim data set was also put through the ANN model to get the final predictions. We do not have access to the results of the aim which leaves the test trial to be our main source for judgement. The measurement which we use for judgement is the F1 score which the avergae of precision and recall and is more suitable when class distribution is uneven. 

![df_results_sc](https://github.com/daistmarco/PredictingPulsarStar/assets/114780077/3ea514b4-7245-4454-8334-9ee72966f9f3)

	
Picture or embedded format for the results table?
Due to all the values being close to on another I question if a graph will bring a benefit in visual as it will be hard to distinguish the difference with labelling everything which could make it feel like noise and cluster it


Do we like the look of the chart? If so I will add more detail to it but it may still look very underwhelming?

![bar_results](https://github.com/daistmarco/PredictingPulsarStar/assets/114780077/0bf8613c-6fdc-46c4-b479-cab5848490ec)

