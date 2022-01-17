import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# read in data
data = pd.read_csv('star_classification.csv')

# split data into training and testing sets
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create dictionary of models
models = {'KNN': KNeighborsClassifier(),
          'Naive Bayes': GaussianNB(),
          'SVM': SVC(),
          'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(),
          'AdaBoost': AdaBoostClassifier(),
          'Gradient Boosting': GradientBoostingClassifier(),
          'Bagging': BaggingClassifier(),
          'Logistic Regression': LogisticRegression(),
          'SGD': SGDClassifier(),
          'MLP': MLPClassifier()}

# create dictionary of models with parameters to tune
params = {'KNN': {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]},
          'Naive Bayes': {},
          'SVM': {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
          'Decision Tree': {'max_depth': [1, 5, 10, 20, 50, 100], 'min_samples_split': [2, 5, 10, 20]},
          'Random Forest': {'n_estimators': [1, 5, 10, 20, 50, 100], 'max_depth': [1, 5, 10, 20, 50, 100], 'min_samples_split': [2, 5, 10, 20]},
          'AdaBoost': {'n_estimators': [1, 5, 10, 20, 50, 100]},
          'Gradient Boosting': {'n_estimators': [1, 5, 10, 20, 50, 100], 'max_depth': [1, 5, 10, 20, 50, 100], 'min_samples_split': [2, 5, 10, 20]},
          'Bagging': {'n_estimators': [1, 5, 10, 20, 50, 100]},
          'Logistic Regression': {'C': [0.1, 1, 10, 100, 1000]},
          'SGD': {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet']},
          'MLP': {'hidden_layer_sizes': [10, 50, 100], 'activation': ['identity', 'logistic', 'tanh', 'relu']}}

# create dictionary of tuned models
tuned_models = {}

# loop through models
for model in models:
    # tune model
    clf = GridSearchCV(models[model], params[model], cv=10)
    # fit model
    clf.fit(X_train, y_train)
    # save model
    tuned_models[model] = clf
    # print results
    print(model)
    print('Best Score: ' + str(clf.best_score_))
    print('Best Parameters: ' + str(clf.best_params_))
    print('\n')

# create dictionary of predictions
predictions = {}

# loop through models
for model in tuned_models:
    # predict classes
    predictions[model] = tuned_models[model].predict(X_test)

# create dictionary of scores
scores = {}

# loop through models
for model in predictions:
    # save accuracy
    scores[model] = accuracy_score(y_test, predictions[model])

# create dictionary of confusion matrices
confusion = {}

# loop through models
for model in predictions:
    # create confusion matrix
    confusion[model] = confusion_matrix(y_test, predictions[model])

# create dataframe of scores
scores = pd.DataFrame(scores.items(), columns=['model', 'score'])

# create dataframe of confusion matrices
confusion = pd.DataFrame(confusion.items(), columns=['model', 'confusion_matrix'])

# plot barplot of scores
sns.barplot(x='model', y='score', data=scores)
plt.show()

# plot confusion matrix heatmap
sns.heatmap(confusion['confusion_matrix'][0], annot=True, cmap='Blues')
plt.show()
