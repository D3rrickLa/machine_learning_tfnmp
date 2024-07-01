# version 2 of the base layout
# first one was straight doo doo



import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split

input_dir = "model/model5/data/dataset4"


"""
GETTING THE DATA / SOME DATA PREPROCESSING
"""
def create_dataframe(input_path):
    data_frames = [] 
    gesture_index = 0
    for file_name in os.listdir(input_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_path, file_name)

            dataframe = pd.read_csv(file_path)

            gesture_action = file_name.split("_")[0]
            gesture_index = int(file_name.split("_")[1].split(".")[0])

            dataframe['gesture'] = gesture_action
            dataframe['gesture_index'] = gesture_index

            dataframe.sort_values(by="frame", inplace=True)
            data_frames.append(dataframe)
    
    if len(data_frames) == 0:
        raise ValueError("dataframe has nothing in it")
    else:
        return data_frames 

df = pd.concat(create_dataframe(input_dir), ignore_index=True)

# inital data cleaning - minimalq
df_cleaned = df.dropna()





# split the dataset
X = df_cleaned.drop(columns=['gesture', 'gesture_index']) # Features - what is needed to cal 'y'
y = df_cleaned['gesture'] # Labels - what we want to calculate

X_lstm = X.reshape(X.shape[0], 1, X.shape[1]) # broken

# split is 60 (train), 20 (val), 20 (test) 
# splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=0)
# further split the training set into a new training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)  # 0.25 x 0.8 = 0.2

"""
GETTING DATA - FEATURE EXTRACTION
    temporal - velocity and acceleration
    spatial - distance between points
"""

def calculate_velocity(data):
    velocity_features = np.diff(data, axis=0)
    return np.vstack([velocity_features, np.zeros((1, velocity_features.shape[1]))]) # vstack - stacks df vertically, the np.zero ensure the array has the same numebr of rows as 'data'

def calculate_acceleration(data):
    acceleration_features = np.diff(data, n=2, axis=0)
    return np.vstack([acceleration_features, np.zeros((2, acceleration_features.shape[1]))])

def calculate_spatial_distance(data): # Euclidean distance between points
    n_points = data.shape[0]

    n_landmarks = data.shape[1]

    spatial_distance_features = np.zeros((n_points, n_landmarks))

    for i in range(n_points):
        for j in range(n_landmarks):
            spatial_distance_features[i, j] = euclidean(data[i], data[j])

    return spatial_distance_features

def extract_features(X, is_training):
    """
    Function to preprocess and extract features from input data X.
    
    Parameters:
    - X: Input data as a numpy array or pandas DataFrame.
    
    Returns:
    - X_velocity: Velocity features after preprocessing.
    - X_acceleration: Acceleration features after preprocessing.
    - X_spatial_distance: Spatial distance features after preprocessing.
    """
    # 1. Normalize the data
    scaler = StandardScaler()
    if(is_training):
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # 2. Calculate features
    X_velocity = calculate_velocity(X_scaled)
    X_acceleration = calculate_acceleration(X_scaled)
    X_spatial_distance = calculate_spatial_distance(X_scaled)

    
    
    return X_velocity, X_acceleration, X_spatial_distance

# Assuming calculate_velocity, calculate_acceleration, and calculate_spatial_distance are defined functions

# Example concatenation and validation
def concatenate_features(X_velocity, X_acceleration, X_spatial_distance):
    """
    Concatenate extracted features if shapes match.
    
    Parameters:
    - X_velocity: Velocity features.
    - X_acceleration: Acceleration features.
    - X_spatial_distance: Spatial distance features.
    
    Returns:
    - Concatenated features (if shapes match), otherwise None.
    """
    if X_velocity.shape[0] == X_acceleration.shape[0] == X_spatial_distance.shape[0]:
        X_features = np.concatenate([X_velocity, X_acceleration, X_spatial_distance], axis=1)
        # print("Concatenated features shapes:")
        # print(X_features.shape)
        return X_features
    else:
        print("\nError: Shapes do not match for concatenation.")
        # Print individual shapes for debugging
        print(X_velocity.shape, X_acceleration.shape, X_spatial_distance.shape)
        return None

# Example usage:
X_train_velocity, X_train_acceleration, X_train_spatial_distance = extract_features(X_train, True)
X_val_velocity, X_val_acceleration, X_val_spatial_distance = extract_features(X_val, False)
X_test_velocity, X_test_acceleration, X_test_spatial_distance = extract_features(X_test, False)


# Concatenate features for train set
X_train_features = concatenate_features(X_train_velocity, X_train_acceleration, X_train_spatial_distance)

# Concatenate features for validation set
X_val_features = concatenate_features(X_val_velocity, X_val_acceleration, X_val_spatial_distance)

# Concatenate features for test set
X_test_features = concatenate_features(X_test_velocity, X_test_acceleration, X_test_spatial_distance)


