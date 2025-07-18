"""
Script to demonstrate decreasing accuracy of model with lower training size...
"""
from updated_classifiers import create_model, obtain_raw_and_vectorized_keys, CONFIG, SCORES_PATH, SCORE_KEY, URL_DATASET, EMBER_DATASET
from sklearn.metrics import accuracy_score
import csv
import pandas as pd

datasets_to_process = [URL_DATASET, EMBER_DATASET]
train_sizes = [1, 0.2, 0.4, 0.6, 0.8]
trials = 5
with open("models/model_degrad_results.csv", "a") as file:
    writer = csv.writer(file)
    writer.writerow(['dataset', 'bytes', 'construct_time', 'train_time', 'accuracy'])
    for dataset in datasets_to_process:
        print(f"training on {dataset}")
        keys, vectorized_keys, labels = obtain_raw_and_vectorized_keys(dataset, create_data=False)
        for trial in range(trials):
            print(f"starting trial {trial}...")
            for train_size in train_sizes:
                print(f"evaluating train set size {train_size}")
                clf, size_in_bytes, construct_time, train_time, accuracy = create_model(keys, vectorized_keys, labels,
                                                                            dataset, save_model=False, train_size=train_size, train_random=None, sample_random=None)
                print("model size: ", size_in_bytes)
                print("construct time: ", construct_time)
                print("train time: ", train_time)
                y_pred = clf.predict(vectorized_keys)
                accuracy = accuracy_score(labels, y_pred)
                print(f"overall accuracy: {accuracy:.4f}")
                # check accuracy on complete dataset instead of just subset of training now...
                writer.writerow([dataset, str(size_in_bytes), train_size, str(construct_time), str(train_time), f'{accuracy:.4f}'])