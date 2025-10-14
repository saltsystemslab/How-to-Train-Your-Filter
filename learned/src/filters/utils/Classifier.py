import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import os
import csv




def train_url_classifier(n_estimators, max_leaves, filename=None, keys=None, labels=None, modelname='model', save_model=False):
    """
    filename should point to a file where one column is the key and the other is the label
    keys should be vectorized
    """
    # first, get the set of keys that we want to train on'
    X = None
    Y = None
    df = None
    if filename is not None:
        df = pd.read_csv(f"data/{filename}.csv")
        X = df['key']
        Y = df['label']
    elif keys is not None:
        X = keys
        Y = labels
    else:
        raise Exception("No valid keys given to classifier")

    # now, split into a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=26)

    # finally, train a random forest classifier with the appropriate parameters
    clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaves, random_state=42)
    clf.fit(X_train, y_train) 

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # we then save the model, returning the model and its size
    if save_model:
        with open('models/' + modelname + '.pkl', 'wb') as f:
            pickle.dump(clf, f)
    return clf

# classifier = train_url_classifier('URL_data', 50, 20)
if __name__ == "__main__":
    # here, we want to train some classifiers then update the scores using those classifiers

    # first, grab the dataset and create a set of keys and labels to train on
    print("TODO")