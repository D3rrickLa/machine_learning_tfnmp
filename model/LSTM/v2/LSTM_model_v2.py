# Version 2 of LSTM model, will be more precise than before
from itertools import combinations
import os
import sys
import time

from keras._tf_keras.keras.callbacks import EarlyStopping 
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from keras._tf_keras.keras.regularizers import l2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.stats import skew, kurtosis
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def create_dataframe_from_data(input_path: str):
    data_frames = []
    landmark_cols = []
    landmark_world_cols = [] 
    gesture_index = 0 

    for file_name in os.listdir(input_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_path, file_name)

            dataframe = pd.read_csv(file_path)

            # Gathers the landmark column names
            if(len(landmark_cols) == 0 and len(landmark_world_cols) == 0):
                landmark_cols = [col for col in dataframe.columns if col.startswith(("x", "y", "z"))]
                landmark_world_cols = [col for col in dataframe.columns if col.startswith(("wx", "wy", "wz"))]
            
            gesture = file_name.split("_")[0]
            gesture_index = int(file_name.split("_")[1].split(".")[0])

            dataframe["gesture"] = gesture
            dataframe["gesture_index"] = gesture_index

            dataframe.sort_values(by="frame", inplace=True)

            data_frames.append(dataframe)

    if len(data_frames) == 0:
        raise ValueError("Dataframe has no data")
    else:
        return pd.concat(data_frames, ignore_index=True), landmark_cols, landmark_world_cols

def split_dataset(dataframe: pd.DataFrame, label_col: str, train_ratio=0.6 , val_ratio=0.2, test_ratio=0.2):

    train_frames = []
    val_frames = []
    test_frames = []

    for _, gesture_data in dataframe.groupby("gesture_index"):
        n_frames = len(gesture_data)
        n_train = int(n_frames * train_ratio)
        n_val = int(n_frames * val_ratio)

        train_split = gesture_data.iloc[:n_train]
        val_split = gesture_data.iloc[n_train:n_train + n_val]
        test_split = gesture_data.iloc[n_train + n_val:]

        train_frames.append(train_split)
        val_frames.append(val_split)
        test_frames.append(test_split)
    
    train_set = pd.concat(train_frames).reset_index(drop=True)
    val_set = pd.concat(val_frames).reset_index(drop=True)
    test_set = pd.concat(test_frames).reset_index(drop=True)

    # Separate X and y
    X_train = train_set.drop(columns=[label_col])
    y_train = train_set[label_col]
    X_val = val_set.drop(columns=[label_col])
    y_val = val_set[label_col]
    X_test = test_set.drop(columns=[label_col])
    y_test = test_set[label_col]

    return X_train, y_train, X_val, y_val, X_test, y_test

def calculate_elapsed_time(df: pd.DataFrame):

    elapsed_lists = []

    for _, gesture_data in df.groupby("gesture_index"):
        avg_frame_rate = np.mean(gesture_data["frame_rate"])

        for i in gesture_data["frame"]:
            elapsed_lists.append(i / avg_frame_rate)
        
    df['elapsed_time'] = elapsed_lists

    return df

def calculate_temporal_features(df: pd.DataFrame, cols: list):
    velocity_cols = [f"velocity_{col}" for col in cols]
    acceleration_cols = [f"acceleration_{col}" for col in cols]
    jerk_cols = [f"jerk_{col}" for col in cols]
    
    for _, gesture_data in df.groupby("gesture_index"): 
        gesture_data = gesture_data.sort_values(by="frame")

        avg_frame_rate = np.mean(gesture_data["frame_rate"])
        time_diffs = gesture_data["frame"].diff().fillna(1) / avg_frame_rate
        
        velocities = gesture_data[cols].diff().div(time_diffs, axis=0).fillna(0)
        accelerations = velocities.diff().div(time_diffs, axis=0).fillna(0)
        jerks = accelerations.diff().div(time_diffs, axis=0).fillna(0)

        df.loc[gesture_data.index, velocity_cols] = velocities.values
        df.loc[gesture_data.index, acceleration_cols] = accelerations.values
        df.loc[gesture_data.index, jerk_cols] = jerks.values

    return df
  
def calculate_temporal_stats(df: pd.DataFrame, cols: list):
    mean_cols = [f"mean_{col}" for col in cols]
    var_cols = [f"variance_{col}" for col in cols] 
    dev_cols = [f"deviation_{col}" for col in cols] 
    skew_cols = [f"skew_{col}" for col in cols] 
    kurt_cols = [f"kurt_{col}" for col in cols] 

    for _, gesture_data in df.groupby("gesture_index"):
        gesture_data = gesture_data.sort_values(by="frame")

        df.loc[gesture_data.index, dev_cols] = gesture_data[cols].rolling(2).std(engine="cython").values # might convert these to numpy for better efificeny in the future
        df.loc[gesture_data.index, var_cols] = gesture_data[cols].rolling(2).var(engine="cython").values
        df.loc[gesture_data.index, skew_cols] = gesture_data[cols].rolling(6).skew().values
        df.loc[gesture_data.index, kurt_cols] = gesture_data[cols].rolling(6).kurt().values
        df.loc[gesture_data.index, mean_cols] = gesture_data[cols].expanding().mean(engine="cython").values

    return df

def calculate_landmark_distances(df: pd.DataFrame, cols: list):
    distance_columns = [f"lm_distance_{i}_{j}" for i in range(len(cols)//3) for j in range(len(cols)//3)]

    for _, gesture_data in df.groupby("gesture_index"):
        gesture_data = gesture_data.sort_values(by="frame")
        
        coords = gesture_data[cols].values.reshape(-1, len(cols) // 3, 3)
        distances = np.sqrt(np.sum((coords[:, :, None] - coords[:, None, :])**2, axis=-1))
        
        # we technically should do something called zero out - basically in the df x_0/x_1 == x_1/x_0 (redundant)

        distances_flat = distances.reshape(-1, len(distance_columns))
        df.loc[gesture_data.index, distance_columns] = distances_flat

    return df

def calculate_landmark_angles(df: pd.DataFrame, cols: list):
    angles_per_gesture_list = []
    for _, gesture_data in df.groupby("gesture_index"):
        gesture_data = gesture_data.sort_values(by="frame")
        gesture_points = gesture_data[cols]
        angles_for_gesture = []

        
        # Iterate over each pair of consecutive points
        for i in range(len(gesture_points) - 1):
            point_a = gesture_points.iloc[i]
            point_b = gesture_points.iloc[i + 1]

            angles = []
            
            # Iterate over each landmark
            for j in range(21):
                idx = j  # Adjust if cols include additional information beyond x, y, z (e.g., wx, wy, wz)
                
                # Extract coordinates for point_a and point_b
                ax, ay, az = point_a[f"x_{idx}"], point_a[f"y_{idx}"], point_a[f"z_{idx}"]
                bx, by, bz = point_b[f"x_{idx}"], point_b[f"y_{idx}"], point_b[f"z_{idx}"]

                # Calculate dot product
                dot_prod = ax * bx + ay * by + az * bz

                # Calculate magnitudes
                magnitude1 = np.linalg.norm([ax, ay, az])
                magnitude2 = np.linalg.norm([bx, by, bz])

                # Calculate angle in degrees
                if magnitude1 > 0 and magnitude2 > 0:
                    angle = np.arccos(np.clip(dot_prod / (magnitude1 * magnitude2), -1.0, 1.0)) * (180 / np.pi)
                else:
                    angle = 0.0  # Handle division by zero or near-zero magnitude cases

                angles.append(angle)

            angles_for_gesture.append(angles)

        angles_per_gesture_list.extend(angles_for_gesture) 

    # Create DataFrame with angles_per_gesture_list
    angles_cols = [f"angle_{n1}" for n1 in range(21)]
    angles_per_gesture_list.insert(0, [0.0] * len(angles_cols))
    angles_df = pd.DataFrame(angles_per_gesture_list, columns=angles_cols)
    # Append angles_df to df
    df = pd.concat([df, angles_df], axis=1)

    return df

def calculate_hand_motion_features(df: pd.DataFrame, landmark_cols: list):
    """
    List of features
        Elasped time - time of the the recorded gesture since frame 0 ✅
        velocity ✅
        acceleration ✅
        jerk ✅
        pairwise distances ✅
        landmark angles ✅
        gesture_stats - mean, variance, skewness, and kurtosis ✅

        process time
            elapsed_time_fuc - 0.15625
            temporal - 6.671875
            stats - 24.1875
            landmarks - 85.421875 -> 32.203125 (more like 60 if running all functions)
            angles - 4.265625

        problems to hand - skew, kurt, and variance have null values - because of the lack of fillna. Skew and kurt are bigger problems cuz of rolling (will use interpolation for this and others)
        distance is just not being calculated  ✅
    """
    df_copy = df.copy()

    s = time.process_time()
    df_elapsed = calculate_elapsed_time(df_copy)    
    print(time.process_time()-s)


    s = time.process_time()
    df_temporal = calculate_temporal_features(df_copy, landmark_cols)
    print(time.process_time()-s)
    
    s = time.process_time()
    df_stats = calculate_temporal_stats(df_copy, landmark_cols)
    print(time.process_time()-s)


    s = time.process_time()
    df_pairwise = calculate_landmark_distances(df_copy, landmark_cols)
    print(time.process_time()-s)


    s = time.process_time()
    df_angle = calculate_landmark_angles(df_copy, landmark_cols)
    print(time.process_time()-s)
   
    
    s = time.process_time()
    df_combined = pd.concat([df_copy, df_angle], axis=1)
    print(time.process_time()-s)
    
    # Ensure there are no duplicate columns
    df_combined = df_combined.loc[:,~df_combined.columns.duplicated()]
   
    return df_combined

def display_null_columns(df: pd.DataFrame):
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    
    result_df = pd.DataFrame({'Column': null_columns.index, 'Null Count': null_columns.values})
    return result_df

def main():
    input_dir = "data/data_2"

    # Step 1: Get data into frame
    dataframe, landmark_cols, landmark_world_cols = create_dataframe_from_data(input_dir)

    # Step 2: Split data into train test val sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(dataframe, "gesture")

    # Step 3: Feature Engineer
    isActive = True
    if os.path.exists("model/LSTM/v2/X_train_fe.csv") and isActive == True:
        X_train_fe = pd.read_csv("model/LSTM/v2/X_train_fe.csv")
        X_val_fe = pd.read_csv("model/LSTM/v2/X_val_fe.csv")
        X_test_fe = pd.read_csv("model/LSTM/v2/X_test_fe.csv")
        print("imported")
    else:
        X_train_fe = calculate_hand_motion_features(X_train, landmark_cols)
        X_val_fe = calculate_hand_motion_features(X_val, landmark_cols)
        X_test_fe = calculate_hand_motion_features(X_test, landmark_cols)

        X_train_fe.to_csv("model/LSTM/v2/X_train_fe.csv", index=False)
        X_val_fe.to_csv("model/LSTM/v2/X_val_fe.csv", index=False)
        X_test_fe.to_csv("model/LSTM/v2/X_test_fe.csv", index=False)

    # Preprocessing 
    # need to have numeric, cat, and ordinal cols
    timeseries_columns = (
        landmark_cols + landmark_world_cols + ['elapsed_time'] + [f"velocity_{col}" for col in landmark_cols] + [f"acceleration_{col}" for col in landmark_cols] 
        + [f"jerk_{col}" for col in landmark_cols]+ [f"mean_{col}" for col in landmark_cols] + [f"variance_{col}" for col in landmark_cols] 
        + [f"deviation_{col}" for col in landmark_cols] + [f"skew_{col}" for col in landmark_cols] + [f"kurt_{col}" for col in landmark_cols] 
        + [f"lm_distance_{i}_{j}" for i in range(len(landmark_cols)//3) for j in range(len(landmark_cols)//3)] + [f"angle_{n1}" for n1 in range(21)] + ["score"]
    )
    print(f"{X_train_fe.shape}||{X_val_fe.shape}||{X_test_fe.shape}")

    numerical_columns = ["frame_rate","frame_width","frame_height","gesture_index"]
    categorical_columns = ['hand', 'gesture_index']
    
    ts_numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('imputer2', SimpleImputer(strategy="mean")),
        ('scaler', MinMaxScaler()),
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ("normalize", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ts_num', ts_numerical_transformer, timeseries_columns),
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='passthrough',
        sparse_threshold=0,
        n_jobs=-1
    )
    # preprocessor.set_output(transform="pandas")
    
    X_train_transformed = preprocessor.fit_transform(X_train_fe)
    X_val_transformed = preprocessor.transform(X_val_fe)
    X_test_transformed = preprocessor.transform(X_test_fe)
    print("done")
    label_encoder = LabelEncoder()
    combined_labels = pd.concat([y_train, y_val, y_test])
    label_encoder.fit(combined_labels)

    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
   
    print(f"{X_train_transformed.shape}||{X_val_transformed.shape}||{X_test_transformed.shape}")
    print(f"{y_train_encoded.shape}||{y_val_encoded.shape}||{y_test_encoded.shape}")

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_transformed, y_train_encoded)
    model_1 = SelectFromModel(lasso, prefit=True)

    X_train_transformed = model_1.transform(X_train_transformed)
    X_val_transformed = model_1.transform(X_val_transformed)
    X_test_transformed = model_1.transform(X_test_transformed)

    print("Selected features (Lasso):", model_1.get_support(indices=True))

    # Reshape the selected features for LSTM input
    X_train_reshaped = X_train_transformed.reshape((X_train_transformed.shape[0], X_train_transformed.shape[1], 1))
    X_val_reshaped = X_val_transformed.reshape((X_val_transformed.shape[0], X_val_transformed.shape[1], 1))
    X_test_reshaped = X_test_transformed.reshape((X_test_transformed.shape[0], X_test_transformed.shape[1], 1))

    print(f"{X_train_transformed.shape}||{X_val_transformed.shape}||{X_test_transformed.shape}")
    print(f"{y_train_encoded.shape}||{y_val_encoded.shape}||{y_test_encoded.shape}")

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(X_train_transformed.shape[1], 1), return_sequences=True, kernel_regularizer=l2()))
    model.add(Dropout(0.2))
    model.add(LSTM(units=8, return_sequences=False)),  # Optional additional LSTM layer
    model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))


    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_reshaped, y_train_encoded, 
        epochs=10, 
        batch_size=32,
        validation_data=(X_val_reshaped, y_val_encoded), 
        callbacks=[early_stopping])

    # Evaluate the model on test set
    test_loss, test_acc = model.evaluate(X_test_reshaped, y_test_encoded)
    print(f'Test Accuracy: {test_acc} || Test Loss: {test_loss}')


        

main()