"""
Script to help choose parameters for each model type that
the learned filters may use.
"""
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from updated_classifiers import obtain_raw_and_vectorized_keys
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
import sys
import time

# create a classifier variant which checks parameter size during the fit
# by breaking early if the resulting saved classifier exceeds the
# defined space budget
class SizeConstrainedClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn wrapper that constrains model size in bytes
    for RandomForest, DecisionTree, and LogisticRegression (with L1) models.
    """

    def __init__(
        self,
        clf_type='random_forest',
        max_bytes=1_000_000,
        # RandomForest / DecisionTree parameters
        n_estimators=100,
        max_depth=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        # LogisticRegression parameters
        penalty='l1',
        solver='saga',
        C=1.0,
        max_iter=2000,
    ):
        self.clf_type = clf_type
        self.max_bytes = max_bytes

        # RandomForest / DecisionTree parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

        # Logistic Regression parameters
        self.penalty = penalty
        self.solver = solver
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        # create a classifier matching the given input
        if self.clf_type == 'random_forest':
            self.base_classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf
            )
        elif self.clf_type == 'decision_tree':
            self.base_classifier = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf
            )
        elif self.clf_type == 'logistic_regression':
            self.base_classifier = LogisticRegression(
                penalty=self.penalty,
                solver=self.solver,
                C=self.C,
                max_iter=self.max_iter
            )
        else:
            raise ValueError(f"Unknown clf_type: {self.clf_type}")

        self.base_classifier.fit(X, y)

        # check the size of the actual model to ensure it fits within
        # the minimum space budget.
        model_size = sys.getsizeof(pickle.dumps(self.base_classifier))
        if model_size > self.max_bytes:
            raise ValueError(
                f"Model size {model_size} bytes exceeds max allowed {self.max_bytes} bytes"
            )
        return self

    def predict(self, X):
        return self.base_classifier.predict(X)

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)

# Map each dataset to the minimal space budget it must fit under.
# These numbers were taken from experiment results using the AdaptiveQF.
datasets = {"url": 69160, "ember": 539890, "shalla": 4280640, "caida": 2144675}

# map each classifier to the parameter grid to search over
models = {'random_forest': { 'n_estimators': [10, 20, 40, 80, 160, 320, 640, 1280], 'max_leaf_nodes': [10, 20, 40, 80, 160, 320, 640, 1280]},
          'decision_tree': { 'max_leaf_nodes': [10, 20, 40, 80, 160, 320, 640, 1280]},
          'logistic_regression': { 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}}

train_size = 0.3

for dataset_name in datasets:
    print(f"Starting parameter search for dataset: {dataset_name}")
    keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset_name)

    # select a subset for training
    pos_rows = (labels == 1)
    neg_rows = (labels == 0)
    pos_keys, pos_vec, pos_labels = keys[pos_rows], vectorized_keys[pos_rows], labels[pos_rows]
    neg_keys, neg_vec, neg_labels = keys[neg_rows], vectorized_keys[neg_rows], labels[neg_rows]
    neg_x_train, neg_x_test, neg_y_train, neg_y_test = train_test_split(neg_vec, neg_labels, train_size=train_size)
    X_train = np.vstack([pos_vec, neg_x_train])
    y_train = np.concatenate([pos_labels, neg_y_train])

    for model_type, param_grid in models.items():
        print(f"Searching parameters for model: {model_type}")
        model = SizeConstrainedClassifier(clf_type=model_type, max_bytes=datasets[dataset_name])
        start = time.perf_counter()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True), n_jobs=-1, scoring='balanced_accuracy', error_score=-1)
        end = time.perf_counter()
        # get the size of the resulting model
        fit_start = time.perf_counter()
        grid_search.fit(X_train, y_train)
        fit_end = time.perf_counter()
        size_in_bytes = sys.getsizeof(pickle.dumps(grid_search.best_estimator_.base_classifier))
        with open(f'../results/parameter_search/results.txt', 'a') as f:
            f.write(f"Dataset: {dataset_name}, Model: {model_type}, Best Params: {grid_search.best_params_}, Size (bytes): {size_in_bytes}, Best Score: {grid_search.best_score_}, Time (seconds): {end - start}, Fit Time (seconds): {fit_end - fit_start}\n")
    print('------------------------')