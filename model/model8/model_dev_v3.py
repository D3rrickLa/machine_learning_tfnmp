# the data formatting might be a problem
import csv
import math
import os  
import time 

from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras._tf_keras.keras.metrics import MeanAbsoluteError, Accuracy, Precision, Recall, MeanSquaredError
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import Adam , RMSprop, Nadam
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Masking, InputLayer, Conv1D, MaxPooling1D, Flatten, TimeDistributed, LayerNormalization, Activation
from keras._tf_keras.keras.regularizers import L1L2, L1, L2
from keras._tf_keras.keras.utils import to_categorical
import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder 

from FeatureEngineering import FeatureEngineering as fe 

def create_dataframe_from_data(input_path: str):
    dataframes = []
    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)

        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data)

        gesture = file_name.split("_")[0]
        gesture_index = int(file_name.split("_")[1].split(".")[0])
        
        # add two new columns to df 
        df["gesture"] = gesture
        df["gesture_index"] = gesture_index

        dataframes.append(df)
        df.sort_values(by="frame", inplace=True)
    
    return pd.concat(dataframes, ignore_index=True) if len(dataframes) > 0 else ValueError("Dataframe is empty")

def create_dict_from_df(df: pd.DataFrame):
    diction = {x: [] for x in np.unique(df["gesture"].values.tolist())}
    for gesture_index, gesture_data in df.groupby("gesture_index"):
        gesture = np.unique(gesture_data["gesture"].values.tolist())[0]
        tmp = diction[gesture] + [gesture_index]
        diction.update({gesture:tmp})
    return diction

def split_dataset(df: pd.DataFrame, target_label: str, additional_targets: list=None, train_ratio=0.7, val_ratio=0.15 ,test_ratio=0.15):
    # probably make this train, val, test
    assert train_ratio + val_ratio + test_ratio == 1.0, "ratios must sum to 1."

    gesture_index_dict = create_dict_from_df(df)

    train_indices = []
    val_indices = []
    test_indices = []

    for _, indices in gesture_index_dict.items():
        n_total = len(indices)
        n_train = math.ceil(n_total * train_ratio)
        n_val = math.ceil(n_total * val_ratio)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    grouped_data = df.groupby("gesture_index")

    train_frames, val_frames, test_frames = [], [], []
    for idx in train_indices:
        train_frames.append(grouped_data.get_group(idx))
        
    for idx in val_indices:
        val_frames.append(grouped_data.get_group(idx))
        
    for idx in test_indices:
        test_frames.append(grouped_data.get_group(idx))

    # Concatenate the dataframes to create the final train and test sets
    train_set = pd.concat(train_frames).reset_index(drop=True)
    val_set = pd.concat(val_frames).reset_index(drop=True)
    test_set = pd.concat(test_frames).reset_index(drop=True)

    # Separate X and y
    X_train = train_set.drop(columns=[target_label])
    y_train = train_set[[target_label] + additional_targets] if additional_targets else train_set[[target_label]]
    X_val = val_set.drop(columns=[target_label])
    y_val = val_set[[target_label] + additional_targets] if additional_targets else val_set[[target_label]]
    X_test = test_set.drop(columns=[target_label])
    y_test = test_set[[target_label] + additional_targets] if additional_targets else test_set[[target_label]]


    return X_train, y_train, X_val, y_val, X_test, y_test

def transform_to_sequences(df: pd.DataFrame, sequence_length):
    sequences = []
    labels = []
    grouped = df.groupby('gesture_index')
    
    for _, group in grouped:
        group = group.sort_values('frame').reset_index(drop=True)
        for i in range(0, len(group), sequence_length):
            sequence = group.iloc[i:i+sequence_length].drop(columns=['gesture_index']).values # gesture_index is dropped because it's more like metadata, no value as a features
            if len(sequence) == sequence_length:
                sequences.append(sequence)
                labels.append(group.iloc[0]['gesture']) 
    return np.array(sequences), np.array(labels) 

def save_sequences_to_csv(sequences, labels, filename, df):
    with open(filename, mode="w", newline='') as file:
        writer = csv.writer(file)
        header = []      
        for i in range(30):
            for col in df.columns[:-2]:  # Exclude 'gesture_index' and 'gesture' columns
                header.append(f"{col}_frame_{i}")
        header.append("gesture")
        writer.writerow(header)
        
        # Write data
        for sequence, label in zip(sequences, labels):
            flattened_sequence = sequence.flatten()
            writer.writerow(np.append(flattened_sequence, label))

input_path = "data/data_3"
dataframe = create_dataframe_from_data(input_path)
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(dataframe, target_label='gesture', additional_targets=[])

# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
# print(X_train["gesture_index"].tail(), X_val["gesture_index"].head())

# X, y = transform_to_sequences(dataframe, 30)

# # print("Transformed Sequences:")
# # print(X)
# # print("Labels:")
# # print(y)

# print(np.array(X).shape)
# print(np.array(y).shape)

# Apply transform_to_sequence
X_train_seq = transform_to_sequences(X_train, 30)
X_val_seq = transform_to_sequences(X_val, 30)
X_test_seq = transform_to_sequences(X_test, 30)

# Check the shapes of the sequences
print(X_train_seq.shape)
print(X_val_seq.shape)
print(X_test_seq.shape)

