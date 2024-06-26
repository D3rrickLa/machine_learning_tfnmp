# version 2 of the base layout
# first one was straight doo doo



import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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

# inital data cleaning - minimal
df_cleaned = df.dropna()





# split the dataset
X = df_cleaned.drop(columns=['gesture', 'gesture_index']) # Features - what is needed to cal 'y'
y = df_cleaned['gesture'] # Labels - what we want to calculate

# split is 60 (train), 20 (val), 20 (test) 
# splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# further split the training set into a new training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)  # 0.25 x 0.8 = 0.2

"""
GETTING DATA - FEATURE EXTRACTION
    temporal - velocity and acceleration
    spatial - distance between points
"""

def calcualte_velocity(data):
    velocity_features = np.diff(data, axis=0)
    return np.vstack([velocity_features, np.zeros((1, velocity_features.shape[1]))]) # vstack - stacks df vertically, the np.zero ensure the array has the same numebr of rows as 'data'

def calculate_acceleration(data):
    acceleration_features = np.diff(data, n=2, axis=0)
    return np.vstack([acceleration_features, np.zeros((2, acceleration_features.shape[1]))])

def calculate_spatial_distance(data): # Euclidean distance between points
    # Calculate pairwise Euclidean distances between points
    spatial_distance_features = cdist(data, data, 'euclidean')
    return spatial_distance_features

def extract_features(X_train, X_val, X_test):
    X_train_velocity = calcualte_velocity(X_train)
    X_val_velocity = calcualte_velocity(X_val)
    X_test_velocity = calcualte_velocity(X_test)

    X_train_acceleration = calculate_acceleration(X_train)
    X_val_acceleration = calculate_acceleration(X_val)
    X_test_acceleration= calculate_acceleration(X_test)

    X_train_spatial_distance = calculate_spatial_distance(X_train)
    X_val_spatial_distance = calculate_spatial_distance(X_val)
    X_test_spatial_distance = calculate_spatial_distance(X_test)

    # Example: Concatenate features for training, validation, and testing sets
    # print(f"{X_train_velocity.shape} || {X_val_velocity.shape} || {X_test_velocity.shape}")
    # print(f"{X_train_acceleration.shape} || {X_val_acceleration.shape} || {X_test_acceleration.shape}")
    # print(f"{X_train_spatial_distance.shape} || {X_val_spatial_distance.shape} || {X_test_spatial_distance.shape}")
    X_train_features = np.concatenate([X_train_velocity, X_train_acceleration, X_train_spatial_distance], axis=1)
    X_val_features = np.concatenate([X_val_velocity, X_val_acceleration, X_val_spatial_distance], axis=1)
    X_test_features = np.concatenate([X_test_velocity, X_test_acceleration, X_test_spatial_distance], axis=1)
    
    return X_train_features, X_val_features, X_test_features

X_train_features, X_val_features, X_test_features = extract_features(X_train, X_val, X_test)



"""
PREPROCESSING
STEPS
    cleaning data - correcting, removing dups
    handle missing data 
    normalizing the data - scaling 
"""

