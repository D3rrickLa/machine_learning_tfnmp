from itertools import combinations
import os

import numpy as np
import pandas as pd


def calculate_temporal_features_per_gesture(df : pd.DataFrame, landmark_cols: list):
    """
    Calculates velocity, acceleration, and jerk for each landmark column in a DataFrame,
    resetting calculations for each new gesture using gesture_index.

    Args:
        df (pd.DataFrame): The DataFrame containing gesture data with frame, gesture_index,
                           gesture, and landmark coordinates (x, y, z) columns.
        landmark_cols (list): A list of column names representing landmark coordinates.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for velocity, acceleration,
                      and jerk of each landmark, resetting calculations at gesture boundaries.
    """

    df_copy = df.copy()
    velocity_cols = [f"velcotiy_{col}" for col in landmark_cols]
    acceleration_cols = [f"acceleration_{col}" for col in landmark_cols]
    jerk_cols = [f"jerk_{col}" for col in landmark_cols]

    df = df.sort_values(by=["frame"])

    for gesture_index, gesture_data in df.groupby('gesture_index'):
        # Reset velocity, acceleration, and jerk for each gesture
        gesture_data[velocity_cols] = 0
        gesture_data[acceleration_cols] = 0
        gesture_data[jerk_cols] = 0

        # Efficient calculation using vectorized operations (if applicable)
        if pd.api.types.is_numeric_dtype(gesture_data[landmark_cols]):
            df[velocity_cols] = df[landmark_cols].diff(fill_value=0)
            df[acceleration_cols] = df[velocity_cols].diff(fill_value=0)
            df[jerk_cols] = df[acceleration_cols].diff(fill_value=0)
        
        # Alt: calculation using a loop (for non-numeric dtypes)
        else:
            for i in range(1, len(gesture_data)):
                for col in landmark_cols:
                    gesture_data.loc[i, f"velocity_{col}"] = gesture_data.loc[i, col] - gesture_data.loc[i - 1, col]
                    gesture_data.loc[i, f"acceleration_{col}"] = gesture_data.loc[i, f"velocity_{col}"] - gesture_data.loc[i - 1, f"velocity_{col}"]
                    gesture_data.loc[i, f"jerk_{col}"] = gesture_data.loc[i, f"acceleration_{col}"] - gesture_data.loc[i - 1, f"acceleration_{col}"]

    return df

def calculate_landmark_distances(df : pd.DataFrame, landmark_cols: list):
    """
    Calculates pairwise Euclidean distances between all landmark combinations
    within each gesture, resetting calculations for each new gesture using gesture_index.

    Args:
        df (pd.DataFrame): The DataFrame containing gesture data with frame, gesture_index,
                           gesture, and landmark coordinates (x, y, z) columns.
        landmark_cols (list): A list of column names representing landmark coordinates.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for pairwise distances
                      between landmark pairs, calculated for each gesture.
    """

    # Iterate over gesture groups using a boolean mask
    for gesture_index, gesture_data in df.groupby("gesture_index"):
        # Sort by frame so calculations within a gesture are consistent
        gesture_data = gesture_data.sort_values(by="frame")

        landmark_pairs = list(combinations(landmark_cols, 2))
        
        for(col1, col2) in landmark_pairs:
            idx1 = col1[1:]
            idx2 = col2[1:]

            if idx1 == idx2:
                continue

            x1, y1, z1 = f'x{idx1}', f'y{idx1}', f'z{idx1}'
            x2, y2, z2 = f'x{idx2}', f'y{idx2}', f'z{idx2}'
            distance_col = f'distance_{idx1}_{idx2}'

            # Calculate distance using vectorized oprerations (if applicable)
            if pd.api.types.is_numeric_dtype(gesture_data[[x1, y1, z1, x2, y2, z2]]):
                
                df[distance_col] = np.sqrt((df[x1] - df[x2])**2 + (df[y1] - df[y2])**2 + (df[z1] - df[z2])**2)  
            
            else:
                raise ValueError("Landmark data types must be numeric for distance calculation")
    
    return df  

def calculate_landmark_angles(df: pd.DataFrame, landmark_cols: list):
    """
    Calculate angles between finger joints or between fingers.
    TODO: find out if the values make sense, are we doing calculation between x1 to x1 at frame 2 or the entire hand frame by frame
    NOTE: THIS IS JUST WRONG WE ARE TAKING THE ENTIRE CSV AND THE NEXT ONE OVER - CONFIRMED VIA PRINTING OUT THE GESTURE DATA
            WE WANT IT SO THAT WE COMPARE THE SAME CSV FRAME N TO FRAME N+1

    NOTE: I honestly don't know if this does what it's suppose to

    Args:
        df (pd.DataFrame): DataFrame containing landmark data.
        landmarks (list): List of tuples containing the landmarks between which angles are to be calculated.
                          Each tuple should contain three elements (landmark1, landmark2, landmark3).

    Returns:
        pd.DataFrame: DataFrame containing the angles between the specified landmarks.
    """

    angles_per_gesture_list = []
 
    for gesture_index, gesture_data in df.groupby("gesture_index"):
        gesture_data = gesture_data.sort_values(by="frame")
        gesture_points = gesture_data[landmark_cols].values
        
        # Calculate direction vectors between consecutive frames (we are using all the landmarks)
        gesture_vectors = np.diff(gesture_points, axis=0)

        if gesture_vectors.shape[0] < 2:
            continue

        # Calculate dot product and magnitudes (assuming 3D points)
        dot_products = np.sum(gesture_vectors[:-1] * gesture_vectors[1:], axis=1)
        magnitude1 = np.linalg.norm(gesture_vectors[:-1], axis=1)
        magnitude2 = np.linalg.norm(gesture_vectors[1:], axis=1)

        # Avoid dividing by zero or near-zero values
        mask = (magnitude1 > 1e-6) & (magnitude2 > 1e-6)
        dot_products[mask] /= (magnitude1[mask] * magnitude2[mask])

        # clip values stop prevent invalid input to arccos 
        dot_products = np.clip(dot_products, -1.0, 1.0)

        frame_angles = np.arccos(dot_products) * (180 / np.pi)

        # create dictionary with angles for each gesture
        angles_df = pd.DataFrame({
            'gesture_index': gesture_index,
            'frame': gesture_data['frame'].iloc[1:len(frame_angles)+1],  # Skip the first frame because diff reduces one row
            'angle': frame_angles
        })
        angles_per_gesture_list.append(angles_df)

    all_angles_df = pd.concat(angles_per_gesture_list, ignore_index=True)
    
    # Merge the angles DataFrame back to the original DataFrame
    df_with_angles = pd.merge(df, all_angles_df, on=['gesture_index', 'frame'], how='left')

    return df_with_angles

def calculate_gesture_stats(df: pd.DataFrame, landmark_cols: list):
    """
    Calculates mean, variance, skewness, and kurtosis across frames for each landmark coordinate.

    Args:
        df (pd.DataFrame): The DataFrame containing gesture data with frame, gesture_index,
                           gesture, and landmark coordinates (x, y, z) columns.
        landmark_cols (list): A list of column names representing landmark coordinates.

    Returns:
        pd.DataFrame: The DataFrame with additional columns containing the calculated statistics
                      for each landmark in each gesture.
    """

    stats_cols = [f"{stat}_{col}" for stat in ["mean", "var", "skew", "kurt"] for col in landmark_cols]
    
    # Group by gesture and calculate statistics for each landmark within each gesture
    gesture_stats = df.groupby(['gesture', 'gesture_index'])[landmark_cols].agg(stats_cols)

    # Reset index to have gesture and gesture_index as separate columns
    gesture_stats = gesture_stats.reset_index()

    return pd.concat([df, gesture_stats], axis=1)


def calculate_hand_motion_features(df : pd.DataFrame, landmark_cols : list):
    df_copy = df.copy()

    temp_features_df = calculate_temporal_features_per_gesture(df_copy, landmark_cols)
    # pairwise_df = calculate_landmark_distances(df_copy, landmark_cols)
    # gest_stats_df = calculate_gesture_stats(df_copy, landmark_cols)
    # angles_df = calculate_landmark_angles(df_copy, landmark_cols)

    # print(temp_features_df.columns.tolist())
    # print(pairwise_df.columns.tolist())
    # print(gest_stats_df.columns.tolist())
    # print(angles_df.columns.tolist())

    return 0

def create_dataframe_from_data(input_path: str):
    """
    Combines multiple CSV files from the input path into one dataset

    Arguments:
        input_path: folder location that contains all the CSV (can use both relative and abs pathing)

    Returns:
        pandas dataframe
    
    Raises:
        ValueError: if the data_frame variable has nothing in it 
    """

    data_frames = [] 
    landmark_cols = [] 
    landmark_world_cols = [] 
    gesture_index = 0 

    for file_name in os.listdir(input_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_path, file_name)

            dataframe = pd.read_csv(file_path)

            # Gathers the landmark column names and stores it into a list (once)
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
        raise ValueError("Dataframe has no data in it.")
    else:
        return pd.concat(data_frames, ignore_index=True), landmark_cols, landmark_world_cols

def main():
    input_dir = "data/"
    dataframe, landmark_cols, landmark_world_cols = create_dataframe_from_data(input_dir)

    calculate_hand_motion_features(dataframe, landmark_cols)
    return 0


main()