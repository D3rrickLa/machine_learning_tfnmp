








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

            # gathers the landmark column hands (once)
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

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)

    stratifcation_factor = y 
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, val_index in splitter.split(X, stratifcation_factor):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)



main()