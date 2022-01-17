# stellar_classifier

A search for the best classifier for the Stellar Classification Dataset
(https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17#)

Winner: Random Forest/Bagging tie. 
Honorable mention: Decision tree, with an average training time of 1.2 seconds.

model	              score     avg_training_time

Naive Bayes	        0.71020   0.04s

SVM               	0.95615   1.2m

KNN	                0.89785   15.8s

Decision Tree	      0.96495   1.2s

Random Forest	      0.97755   16.7s

AdaBoost	          0.61105   8.8s

Gradient Boosting	  0.97595   1.2m

Bagging	            0.97730   6.7s

Logistic Regression	0.95385   1.9s

SGD	                0.91195   0.2s

MLP               	0.97025   42s


I provided a commented out grid search dict with some example parameters which can be trimmed and then uncommented, but be sure to only optimize models you're interested in to save time. 
