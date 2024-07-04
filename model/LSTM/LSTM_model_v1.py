








import os

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split, StratifiedShuffleSplit


def calculate_velocity(df):
    return 0

def calculate_hand_motion_features(df, landmark_cols):
    """
    Feature engineers these new features: 
        - normalized landmarks locations 
        - velocity (temporal feature)
        - acceleration (temporal feature)
        - relative pairwise distances (Euclidean distance)
        - relative landmark angles
    """
    return 0

def create_dataframe_from_data(input_path):
    """
    Takes in the multiple CSV files from the input_path and creates 1 dataframe out of them.

    returns: Dataframe
    """

    data_frames = [] 
    landmark_cols = []
    landmark_world_cols = []
    gesture_index = 0 

    for file_name in os.listdir(input_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_path, file_name)

            dataframe = pd.read_csv(file_path)

            # Gathers the landmark column hands (once)
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
        return pd.concat(data_frames, ignore_index=True)


def main():
    input_dir = "data/"
    dataframe = create_dataframe_from_data(input_dir).dropna()

    X = dataframe.drop(columns=['gesture'], axis=1)
    y = dataframe['gesture']

    tscv = TimeSeriesSplit(n_splits=5) # Extracts al lunique value sfrom the gesture_index

    # Identify unique gestures
    unique_gestures = dataframe['gesture_index'].unique()

    for train_gesture_index, test_gesture_index in tscv.split(unique_gestures):
        train_gestures = unique_gestures[train_gesture_index]
        test_gestures = unique_gestures[test_gesture_index]

        # Split data based on gesture indices
        X_train = X[X['gesture_index'].isin(train_gestures)]
        y_train = y[X['gesture_index'].isin(train_gestures)]
        X_test = X[X['gesture_index'].isin(test_gestures)]
        y_test = y[X['gesture_index'].isin(test_gestures)]

        # Optionally split further into validation set (replace 0.8 with your desired ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)

        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)

main()