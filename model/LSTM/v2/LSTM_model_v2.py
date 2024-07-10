# Version 2 of LSTM model, will be more precise than before
from itertools import combinations
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

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
            elapsed_time = i / avg_frame_rate
            elapsed_lists.append(elapsed_time)
        
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

        rolling_dev =  gesture_data[cols].rolling(2).std().fillna(0)
        rolling_var = gesture_data[cols].rolling(2).var()
        rolling_skew = gesture_data[cols].rolling(6).skew()
        rolling_kurt = gesture_data[cols].rolling(6).kurt()
        cumulative_mean = gesture_data[cols].expanding().mean()

        df.loc[gesture_data.index, dev_cols] = rolling_dev.values
        df.loc[gesture_data.index, var_cols] = rolling_var.values
        df.loc[gesture_data.index, skew_cols] = rolling_skew.values
        df.loc[gesture_data.index, kurt_cols] = rolling_kurt.values
        df.loc[gesture_data.index, mean_cols] = cumulative_mean.values

    return df

def calculate_landmark_distances(df: pd.DataFrame, col: list):
    
    df_copy = df.copy()
    landmark_pairs = list(combinations(col, 2))
    
    distance_cols = [f"distance_{idx1}_{idx2}" for idx1, idx2 in landmark_pairs]
    new_columns = []
    
    for _, gesture_data in df_copy.groupby("gesture_index"):
        gesture_data = gesture_data.sort_values(by="frame")
        gesture_distances = []

        for(col1, col2) in landmark_pairs:
            idx1 = col1[1:]
            idx2 = col2[1:]

            x1, y1, z1 = f'x{idx1}', f'y{idx1}', f'z{idx1}'
            x2, y2, z2 = f'x{idx2}', f'y{idx2}', f'z{idx2}'

            if all(pd.api.types.is_numeric_dtype(gesture_data[col]) for col in [x1, y1, z1, x2, y2, z2]): 
                distances = np.sqrt((gesture_data[x1] - gesture_data[x2])**2 + (gesture_data[y1] - gesture_data[y2])**2 + (gesture_data[z1] - gesture_data[z2])**2)
            
            else:
                raise ValueError("Landmark data types must be numeric for distance calculation")

            # Store distances in a list
            gesture_distances.append(distances)

        # Concatenate all distances into a DataFrame
        gesture_distances_df = pd.DataFrame(np.array(gesture_distances).T, columns=distance_cols, index=gesture_data.index)
        new_columns.append(gesture_distances_df)
        
    
    df_copy = pd.concat([df_copy] + new_columns, axis=1)

    return df_copy

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
        landmark angles
        gesture_stats - mean, variance, skewness, and kurtosis ✅
    """
    df_copy = df.copy()

    # df_elapsed = calculate_elapsed_time(df_copy) 
    # df_temporal = calculate_temporal_features(df_copy, landmark_cols)
    # df_stats = calculate_temporal_stats(df_copy, landmark_cols)
    # df_pairwise = calculate_landmark_distances(df_copy, landmark_cols)

    df_angle = calculate_landmark_angles(df_copy, landmark_cols)

    # print(df_copy.columns.values.tolist())

   

    return df_copy

def main():
    input_dir = "data/data_2"

    # Step 1: Get data into frame
    dataframe, landmark_cols, landmark_world_cols = create_dataframe_from_data(input_dir)

    # Step 2: Split data into train test val sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(dataframe, "gesture")

    # Step 3: Feature Engineer
    X_train_fe = calculate_hand_motion_features(X_train, landmark_cols)

main()