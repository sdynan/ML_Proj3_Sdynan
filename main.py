# Sean Dynan
# Machine Learning Project 3
# This program will use random forests to see whether different spots in the ionosphere are good or bad

import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def randomForest():
    df = pd.read_csv("ionosphere.data", sep=",")
    df.sample(5, random_state=44)
    x = df.iloc[:, : -1]  # accessing the lines that read the data points across the x axis
    y = df.iloc[:, -1:] # accessing the data points and dropping last column to get rid of g and b column
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5) # splitting data 50/50
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, oob_score=True)  # using scikit learn random forest classifier
    rf.fit(X_train, y_train.values.ravel()) # fitting model to best learn what outcome will be
    predictions = rf.predict(X_test) # making predictions on x axis of test data
    print(confusion_matrix(y_test, predictions)) # showing accuracy of results
    print('\n')
    print(classification_report(y_test, predictions))  # showing accuracy of predictions




def main():
    randomForest()


if __name__ == '__main__':
    main()
