## Predicting Pulsar Stars

The project predicting pulsar stars comes from a competition from the organisation Kaggle which has already taken place before starting this project. The project files and the machine learning under taken were all done using Python and presented in Notebooks. Due to many Neutron Stars there is a lot of data that would need to be individualy tested to see if it also a Pulsar Star, with generating predictions focus can be put on the potential Pulsar Stars first. The following information is provided from the challenge page:


### Description
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
robert.lyon '@' manchester.ac.uk
From <https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate> 


### Terminology of data explained:
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

IMAGE------------------------------------


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
The visuals below show the final results from all three different algorithms with the Artificial Neural Network with polynomial features along with Principal Component Analysis was found to return the best results in predicting pulsar stars. 
