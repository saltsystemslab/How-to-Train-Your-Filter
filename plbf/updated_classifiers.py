
from ember_import.ember.ember import read_metadata, create_metadata, create_vectorized_features, read_vectorized_features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from src.PLBFs.utils.url_classifier import vectorize_url
import pickle
import sys
import os
import time
import csv

EMBER_DATASET = "ember"
URL_DATASET = "url"
DEFAULT = "default"

SCORES_PATH = "scores_path"
SOURCE_PATH = "dataset_path"
ESTIMATORS = "n_estimators"
LEAVES = "max_leaves"
OBJ_KEY = "object_key"
LABEL_KEY = "label_key"
SCORE_KEY = "score_key"
POS_INDICATOR = "pos_indicator"

CONFIG = {
    DEFAULT: {
        LABEL_KEY: "label",
        SCORE_KEY: "score"
    },
    URL_DATASET: {
        SCORES_PATH: "data/malicious_url_scores.csv",
        SOURCE_PATH: "data/malicious_url_scores.csv",
        OBJ_KEY: "url",
        LABEL_KEY: "type",
        SCORE_KEY: "prediction_score",
        POS_INDICATOR: 1,
        ESTIMATORS: 75,
        LEAVES: 20,
    },
    EMBER_DATASET: {
        SCORES_PATH: "data/combined_ember_metadata.csv",
        SOURCE_PATH: "data/ember",
        OBJ_KEY: "sha256",
        LABEL_KEY: "label",
        SCORE_KEY: "score",
        POS_INDICATOR: 1,
        ESTIMATORS: 120,
        LEAVES: 50
    }
}

def get_ember_keys(create_data=False):
    """
    Obtains a list of vectorized positive and negative keys from the ember dataset.
    """
    # first, create the data if necessary
    if create_data:
        create_metadata(CONFIG[EMBER_DATASET][SOURCE_PATH])
        create_vectorized_features(CONFIG[EMBER_DATASET][SOURCE_PATH], feature_version=1)

    # next, do an initial read of the objects
    meta_df = read_metadata(CONFIG[EMBER_DATASET][SOURCE_PATH])
    x_train, y_train, x_test, y_test = read_vectorized_features(CONFIG[EMBER_DATASET][SOURCE_PATH], feature_version=1)
    

    total_x = np.vstack([x_train, x_test])
    total_y = np.concatenate([y_train, y_test])

    
    # filter out files where there is no label
    not_unlabeled_rows = (total_y != -1)
    result_meta = meta_df[not_unlabeled_rows]
    result_x = total_x[not_unlabeled_rows]
    result_y = total_y[not_unlabeled_rows]
    print("number of labeled rows: ", len(result_x))

    # important - the meta_df has train and test rows aligned with the vectorized features.
    # we can tell because of how creating metadata and vectorized features uses the same
    # raw features, but also because the example notebook in 'resources' runs an example
    # where index is used to connect the rows.
    return np.array(result_meta[CONFIG[EMBER_DATASET][OBJ_KEY]]), result_x, result_y

def obtain_raw_and_vectorized_keys(dataset: str, create_data=False):
    """
    Returns keys, vectorized form, and labels for the specified dataset.

    Usually, keys are used to insert into the filter, but the vectorized keys
    are used for the model to assess the key.
    """
    if dataset == URL_DATASET:
        data = pd.read_csv(CONFIG[URL_DATASET][SOURCE_PATH])
        data[CONFIG[DEFAULT][LABEL_KEY]] = data[CONFIG[URL_DATASET][LABEL_KEY]].apply(lambda x: 1 if x == 'malicious' else 0)
        keys = np.array(data[CONFIG[URL_DATASET][OBJ_KEY]])
        vectorized_keys = np.array([vectorize_url(url) for url in keys])
        labels = np.array(data[CONFIG[DEFAULT][LABEL_KEY]])
        return keys, vectorized_keys, labels
    elif dataset == EMBER_DATASET:
        return get_ember_keys(create_data=create_data)
    else:
        raise Exception(f"{dataset} dataset is not implemented yet.")

def create_model(keys, vectorized_keys, labels, dataset, train_size=0.3, 
                 sample_random=26, train_random=42, save_model=False):
    """
    Creates a model for the given dataset using set configuration parameters.
    """
    n_estimators = CONFIG[dataset][ESTIMATORS]
    max_leaves = CONFIG[dataset][LEAVES]
    pos_rows = (labels == 1)
    neg_rows = (labels == 0)

    pos_keys, pos_vec, pos_labels = keys[pos_rows], vectorized_keys[pos_rows], labels[pos_rows]
    neg_keys, neg_vec, neg_labels = keys[neg_rows], vectorized_keys[neg_rows], labels[neg_rows]

    print("splitting train set")
    neg_x_train, neg_x_test, neg_y_train, neg_y_test = train_test_split(neg_vec, neg_labels, train_size=train_size, random_state=sample_random)

    X_train = np.vstack([pos_vec, neg_x_train])
    y_train = np.concatenate([pos_labels, neg_y_train])
    # then, train a random forest classifier with the appropriate parameters
    print("constructing and training model")
    construct_start = time.time()
    clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaves, random_state=train_random)
    construct_end = time.time()
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_end = time.time()  

    y_pred = clf.predict(neg_x_test)
    accuracy = accuracy_score(neg_y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    # we then save the model, returning the model and its size
    if save_model:
        with open(f'models/{dataset}_{n_estimators}_{max_leaves}.pkl', 'wb') as f:
            pickle.dump(clf, f)

    # get the size of the model
    size_in_bytes = sys.getsizeof(pickle.dumps(clf))
    construct_time = construct_end - construct_start
    train_time = train_end - train_start
    return clf, size_in_bytes, construct_time, train_time, accuracy

def read_model(dataset):
    n_estimators = CONFIG[dataset][ESTIMATORS]
    max_leaves = CONFIG[dataset][LEAVES]
    # first, read the model
    model = None
    with open(f'models/{dataset}_{n_estimators}_{max_leaves}.pkl', 'rb') as f:
        model = pickle.load(f)
    # next, go to the model training data and collect the training time info
    training_df = pd.read_csv("models/model_training_data.csv")
    dataset_row = (training_df['dataset'] == dataset)
    match = training_df.loc[dataset_row].iloc[0]
    bytes = match['bytes']
    construct_time = match['construct_time']
    train_time = match['train_time']
    accuracy = match['accuracy']
    return model, bytes, construct_time, train_time, accuracy

if __name__ == "__main__":
    datasets_to_process = [URL_DATASET, EMBER_DATASET]
    with open("models/model_training_data.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'bytes', 'construct_time', 'train_time', 'accuracy'])
        for dataset in datasets_to_process:
            print(f"training on {dataset}")
            keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset, create_data=False)
            clf, size_in_bytes, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels,
                                                                        dataset, save_model=True)
            print("model size: ", size_in_bytes)
            print("construct time: ", construct_time)
            print("train time: ", train_time)
            writer.writerow([dataset, str(size_in_bytes), str(construct_time), str(train_time), f'{accuracy:.4f}'])

            print("writing to scores...")
            # now update the scores for the dataset (used only for quick results sanity checks)
            # this csv pairs each object/label with its score
            scores_df = pd.read_csv(CONFIG[dataset][SCORES_PATH])
            # find the vector corresponding to the object of each row in the csv then replace the score
            # note that this assumes that the rows are aligned.
            scores_df[CONFIG[dataset][SCORE_KEY]] = clf.predict_proba(vectorized_keys)[:,1]
            scores_df.to_csv(CONFIG[dataset][SCORES_PATH], index=False)




# # create the training and testing metadata
# create_metadata("data/ember")

# print("Obtaining data")
# # use the Ember imported code base to convert downloaded Ember json into training and testing sets
# x_train, y_train, x_test, y_test = read_vectorized_features("data/ember", feature_version=1)

# # Remove all the unlabelled rows (only found in the training data)
# train_rows = (y_train != -1)
# malicious_rows = (y_test == 1)

# malicious_y_test = y_test[malicious_rows]
# malicious_x_test = x_test[malicious_rows]

# # combine the training and testing set into one overall dataset.
# # as described in PLBF, the training set consists of the original training set
# # (without unlabelled files) combined with the malicious files from the test set
# total_x_train = np.vstack((x_train[train_rows], malicious_x_test))
# total_y_train = np.concatenate((y_train[train_rows], malicious_y_test))

# print("Training")
# # Initialize a classifier
# n_estimators = 60
# max_leaves = 20
# modelname = 'ember_model'
# clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaves, random_state=42)
# classifier = clf.fit(total_x_train, total_y_train) # Have the model learn from the training data

# # check the resulting accuracy of the classifier
# y_pred = clf.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

# # the size of the classifier can be found in the pickle file
# with open('models/' + modelname + '.pkl', 'wb') as f:
#     pickle.dump(classifier, f)
# size_bytes = os.path.getsize('models/' + modelname + '.pkl')
# print(size_bytes)

# train_x_classify = pd.read_csv("data/ember_vectorized_x_train_rm_unlabel.csv", index_col=0)
# scores = clf.predict_proba(train_x_classify)[:, 1]
# train_metadata_no_unlabel = pd.read_csv("data/train_metadata_rm_unlabel.csv")
# train_metadata_no_unlabel["score"] = scores

# test_scores = clf.predict_proba(x_test)[:, 1]
# test_metadata = pd.read_csv("data/ember/test_metadata.csv")
# test_metadata["score"] = test_scores

# # create a unified df
# combined_metadata = pd.concat([train_metadata_no_unlabel, test_metadata], ignore_index=True)
# combined_metadata.to_csv("combined_ember_metadata.csv")