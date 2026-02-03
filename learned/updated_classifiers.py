"""
Manages the parameters used to train models for each dataset, and also defines how to read the datasets and perform training.
If run as a main script, edits dataset csv's to also have a column for the key's updated score.
"""
from ember_import.ember.ember import read_metadata, create_metadata, create_vectorized_features, read_vectorized_features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.filters.utils.url_classifier import vectorize_url
import pickle
import sys
import time
import csv

EMBER_DATASET = "ember"
URL_DATASET = "url"
SHALLA_DATASET = "shalla"
CAIDA_DATASET = "caida"
DEFAULT = "default"

PATH = "path"
DATA_PATH = "data_path"
SCORES_PATH = "scores_path"
SOURCE_PATH = "dataset_path"
ESTIMATORS = "n_estimators"
LEAVES = "max_leaves"
OBJ_KEY = "object_key"
LABEL_KEY = "label_key"
SCORE_KEY = "score_key"
POS_INDICATOR = "pos_indicator"
DEC_TREE_NODES = "decision_tree_nodes"
LOGISTIC_REGRESSION_C = "logistic_regression_c"

DATA_PATH = "../data/"

SUPPORTED_MODELS = ["random_forest", "decision_tree", "logistic_regression"]

# updated parameters after running grid search
CONFIG = {
    DEFAULT: {
        LABEL_KEY: "label",
        SCORE_KEY: "score",
    },
    URL_DATASET: {
        PATH: "malicious_url_scores.csv",
        OBJ_KEY: "url",
        LABEL_KEY: "type",
        SCORE_KEY: "prediction_score",
        POS_INDICATOR: 1,
        ESTIMATORS: 30,
        LEAVES: 10,
        DEC_TREE_NODES: 320,
        LOGISTIC_REGRESSION_C: 0.1
    },
    EMBER_DATASET: {
        PATH: "combined_ember_metadata.csv",
        SOURCE_PATH: "data/ember",
        OBJ_KEY: "sha256",
        LABEL_KEY: "label",
        SCORE_KEY: "score",
        POS_INDICATOR: 1,
        ESTIMATORS: 10,
        LEAVES: 320,
        DEC_TREE_NODES: 1280,
        LOGISTIC_REGRESSION_C: 0.00001
    },
    SHALLA_DATASET: {
        PATH: "shalla_combined.csv",
        OBJ_KEY: "url",
        LABEL_KEY: "label",
        SCORE_KEY: "score",
        POS_INDICATOR: 1,
        ESTIMATORS: 20,
        LEAVES: 1280,
        DEC_TREE_NODES: 1280,
        LOGISTIC_REGRESSION_C: 10
    },
    CAIDA_DATASET: {
        PATH: "caida.csv",
        OBJ_KEY: "No.",
        LABEL_KEY: "Label",
        SCORE_KEY: "score",
        POS_INDICATOR: 1,
        ESTIMATORS: 10,
        LEAVES: 1280,
        DEC_TREE_NODES: 1280,
        LOGISTIC_REGRESSION_C: 0.00001
    }
}

def get_ember_keys(create_data=False):
    """
    Obtains a list of vectorized positive and negative keys from the ember dataset.

    Parameters
    ----------
    create_data : bool
        whether or not the ember dataset has previously been built, necessary for metadata and feature setup.
        
    Returns
    -------
    meta_df: np.array
        array of metadata strings
    result_x: np.array
        array of vectorized features
    result_y: np.array
        array of malware classifications
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

    Parameters
    ----------
    dataset : str
        which dataset (i.e. url, ember, shalla, caida) the keys and features should be drawn from
    create_data : bool
        whether or not the ember dataset has previously been built, necessary for metadata and feature setup.
        
    Returns
    -------
    keys: np.array
        array of keys to insert
    vectorized_keys: np.array
        array of vectorized features
    labels: np.array
        array of malware classifications
    """
    if dataset == URL_DATASET:
        data = pd.read_csv(f"{DATA_PATH}{CONFIG[URL_DATASET][PATH]}")
        data[CONFIG[DEFAULT][LABEL_KEY]] = data[CONFIG[URL_DATASET][LABEL_KEY]].apply(lambda x: 1 if x == 'malicious' else 0)
        keys = np.array(data[CONFIG[URL_DATASET][OBJ_KEY]])
        vectorized_keys = np.array([vectorize_url(url) for url in keys])
        labels = np.array(data[CONFIG[DEFAULT][LABEL_KEY]])
        return keys, vectorized_keys, labels
    elif dataset == SHALLA_DATASET:
        data = pd.read_csv(f"{DATA_PATH}{CONFIG[SHALLA_DATASET][PATH]}")
        data[CONFIG[DEFAULT][LABEL_KEY]] = data[CONFIG[SHALLA_DATASET][LABEL_KEY]]
        keys = np.array(data[CONFIG[SHALLA_DATASET][OBJ_KEY]])
        vectorized_keys = np.array([vectorize_url(url) for url in keys])
        labels = np.array(data[CONFIG[DEFAULT][LABEL_KEY]])
        return keys, vectorized_keys, labels
    elif dataset == EMBER_DATASET:
        return get_ember_keys(create_data=create_data)
    elif dataset == CAIDA_DATASET:
        data = pd.read_csv(f"{DATA_PATH}{CONFIG[CAIDA_DATASET][PATH]}")
        # 'index' column is the item we'll insert
        # 'label' column is labels
        # everything else is the vectorized key.
        keys = data[CONFIG[CAIDA_DATASET][OBJ_KEY]]
        keys = np.array(keys, dtype=object)
        labels = np.array(data[CONFIG[CAIDA_DATASET][LABEL_KEY]])
        vectorized_keys = np.array(data.drop(columns=[CONFIG[CAIDA_DATASET][OBJ_KEY], CONFIG[CAIDA_DATASET][LABEL_KEY]]))
        return keys, vectorized_keys, labels
    else:
        raise Exception(f"{dataset} dataset is not supported yet.")

def create_model(keys, vectorized_keys, labels, dataset, train_size=0.3, 
                 sample_random=26, train_random=42, save_model=False, n_estimators=None, max_leaves=None, 
                 model_type="random_forest"):
    """
    Creates a model for the given dataset using set configuration parameters.
    """
    n_estimators = CONFIG[dataset][ESTIMATORS] if n_estimators is None else n_estimators
    max_leaves = CONFIG[dataset][LEAVES] if max_leaves is None else max_leaves
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
    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=CONFIG[dataset][ESTIMATORS], max_leaf_nodes=CONFIG[dataset][LEAVES], random_state=train_random)
    elif model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_leaf_nodes=CONFIG[dataset][DEC_TREE_NODES], random_state=train_random)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(C=CONFIG[dataset][LOGISTIC_REGRESSION_C], max_iter=2000, random_state=train_random)
    else:
        raise Exception(f"Model type {model_type} not supported.")
    construct_end = time.time()
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_end = time.time()  

    total_y_pred = clf.predict(vectorized_keys)
    total_accuracy = accuracy_score(total_y_pred, labels)

    # we then save the model, returning the model and its size
    id = time.time()
    if save_model:
        with open(f'models/{dataset}_{n_estimators}_{max_leaves}_{model_type}_{id}.pkl', 'wb') as f:
            pickle.dump(clf, f)

    # get the size of the model
    size_in_bytes = sys.getsizeof(pickle.dumps(clf))
    construct_time = construct_end - construct_start
    train_time = train_end - train_start
    return clf, size_in_bytes, construct_time, train_time, total_accuracy

def read_model(dataset, model_type="random_forest"):
    n_estimators = CONFIG[dataset][ESTIMATORS]
    max_leaves = CONFIG[dataset][LEAVES]
    # first, read the model
    model = None
    with open(f'models/{dataset}_{n_estimators}_{max_leaves}_{model_type}.pkl', 'rb') as f:
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
    # saves model scores as an additional column of the dataset csvs,
    # useful for plotting the difference between positive and negative key scores.
    datasets_to_process = [URL_DATASET, EMBER_DATASET, SHALLA_DATASET, CAIDA_DATASET]
    with open("models/model_training_data.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'model', 'bytes', 'construct_time', 'train_time', 'accuracy'])
        for dataset in datasets_to_process:
            print(f"training on {dataset}")
            keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset, create_data=False)
            for model_type in SUPPORTED_MODELS:
                if model_type != "logistic_regression":
                    continue
                print(f"using model type: {model_type}")
                clf, size_in_bytes, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels,
                                                                            dataset, model_type=model_type, save_model=True)
                print("model size: ", size_in_bytes)
                print("construct time: ", construct_time)
                print("train time: ", train_time)
                print("model accuracy: ", accuracy)
                writer.writerow([dataset, model_type, str(size_in_bytes), str(construct_time), str(train_time), f'{accuracy:.4f}'])

                print("writing to scores...")
                # now update the scores for the dataset (used only for quick results sanity checks)
                # this csv pairs each object/label with its score
                scores_df = pd.read_csv(f"{DATA_PATH}{CONFIG[dataset][PATH]}")
                # find the vector corresponding to the object of each row in the csv then replace the score
                # note that this assumes that the rows are aligned.
                scores_df[CONFIG[dataset][SCORE_KEY]] = clf.predict_proba(vectorized_keys)[:,1]
                scores_df.to_csv(f"{DATA_PATH}{model_type}_{CONFIG[dataset][PATH]}", index=False)