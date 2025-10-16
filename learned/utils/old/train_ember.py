
from ember_import.ember.ember import read_metadata, create_metadata, create_vectorized_features, read_vectorized_features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import csv

def get_ember_keys(create_data=False):
    """
    Obtains a list of vectorized positive and negative keys from the ember dataset.
    """
    # first, create the data if necessary
    if create_data:
        create_metadata("data/ember")
        create_vectorized_features("data/ember", feature_version=1)

    # next, do an initial read of the objects
    meta_df = read_metadata("data/ember")
    x_train, y_train, x_test, y_test = read_vectorized_features("data/ember", feature_version=1)
    

    total_x = np.vstack([x_train, x_test])
    total_y = np.vstack([y_train, y_test])

    
    # filter out files where there is no label

    # important - the meta_df has train and test rows aligned with the vectorized features.
    # we can tell because of how creating metadata and vectorized features uses the same
    # raw features, but also because the example notebook in 'resources' runs an example
    # where index is used to connect the rows.
    return np.array(meta_df['sha256']), total_x, total_y
    
    # according to the Ember paper, 0 is benign, 1 is malicious, -1 is unlabeled
    # malicious_rows = (total_y == 1)
    # benign_rows = (total_y == 0)
    # pos_meta = meta_df[malicious_rows]
    # pos_x = total_x[malicious_rows]
    # pos_y = total_y[malicious_rows]
    # neg_meta = meta_df[benign_rows]
    # neg_x = total_x[benign_rows]
    # neg_y = total_y[benign_rows]
    # return np.array(pos_meta['sha256']), pos_x, pos_y, np.array(neg_meta['sha256']), neg_x, neg_y
# create the training and testing metadata
create_metadata("data/ember")

print("Obtaining data")
# use the Ember imported code base to convert downloaded Ember json into training and testing sets
x_train, y_train, x_test, y_test = read_vectorized_features("data/ember", feature_version=1)

# Remove all the unlabelled rows (only found in the training data)
train_rows = (y_train != -1)
malicious_rows = (y_test == 1)

malicious_y_test = y_test[malicious_rows]
malicious_x_test = x_test[malicious_rows]

# combine the training and testing set into one overall dataset.
# as described in PLBF, the training set consists of the original training set
# (without unlabelled files) combined with the malicious files from the test set
total_x_train = np.vstack((x_train[train_rows], malicious_x_test))
total_y_train = np.concatenate((y_train[train_rows], malicious_y_test))

print("Training")
# Initialize a classifier
n_estimators = 60
max_leaves = 20
modelname = 'ember_model'
clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaves, random_state=42)
classifier = clf.fit(total_x_train, total_y_train) # Have the model learn from the training data

# check the resulting accuracy of the classifier
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# the size of the classifier can be found in the pickle file
with open('models/' + modelname + '.pkl', 'wb') as f:
    pickle.dump(classifier, f)
size_bytes = os.path.getsize('models/' + modelname + '.pkl')
print(size_bytes)

train_x_classify = pd.read_csv("data/ember_vectorized_x_train_rm_unlabel.csv", index_col=0)
scores = clf.predict_proba(train_x_classify)[:, 1]
train_metadata_no_unlabel = pd.read_csv("data/train_metadata_rm_unlabel.csv")
train_metadata_no_unlabel["score"] = scores

test_scores = clf.predict_proba(x_test)[:, 1]
test_metadata = pd.read_csv("data/ember/test_metadata.csv")
test_metadata["score"] = test_scores

# create a unified df
combined_metadata = pd.concat([train_metadata_no_unlabel, test_metadata], ignore_index=True)
combined_metadata.to_csv("combined_ember_metadata.csv")