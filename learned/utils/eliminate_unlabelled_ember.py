"""
Helper script to pre-process the Ember dataset. The Ember dataset includes unlabelled data
which can't be used for false-positive tests, so we remove those.
"""
import pandas as pd

# Here we create a new version of the test sets without the unlabelled files :)
train_metadata_df = pd.read_csv('data/ember/train_metadata.csv')
vectorized_x_train_df = pd.read_csv('data/ember_data/ember_vectorized_x_train.csv')
vectorized_y_train_df = pd.read_csv('data/ember_data/ember_vectorized_y_train.csv')

# Create a mask for labeled data
labeled_mask = train_metadata_df['label'] != -1

# Filter all in one step
metadata_df = train_metadata_df[labeled_mask]
vectorized_x_train_df = vectorized_x_train_df[labeled_mask]
vectorized_y_train_df = vectorized_y_train_df[labeled_mask]

# Save cleaned versions
metadata_df.to_csv("train_metadata_rm_unlabel.csv", index=False)
vectorized_x_train_df.to_csv("ember_vectorized_x_train_rm_unlabel.csv", index=False)
vectorized_y_train_df.to_csv("ember_vectorized_y_train_rm_unlabel.csv", index=False)