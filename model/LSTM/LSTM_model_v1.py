








from itertools import combinations
import os

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer


def normalize_landmark_locations(df : pd.DataFrame, landmark_cols):
    """
    Normalizes landmark locations in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing landmark data.
        landmark_cols (list): A list of column names representing landmark coordinates.

    Returns:
        pd.DataFrame: The modified DataFrame with normalized landmark columns.

    Raises:
        ValueError: If any column in landmark_cols is not present in the DataFrame.
    """

    for col in landmark_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

        # Calculate mean and standard deviation for the current landmark column
        mean_col = df[col].mean()
        std_col = df[col].std()

        # Update the existing column with normalized values
        df[col] = (df[col] - mean_col) / std_col

    return df

def calculate_landmark_angles(df : pd.DataFrame, landmark_cols):
    angles = {}
    for i in range(len(landmark_cols) - 2):
        col1, col2, col3 = landmark_cols[i], landmark_cols[i+1], landmark_cols[i+2]
        angle_col = f"angle_{col1}_{col2}_{col3}"
        
        # Compute vectors between landmarks
        vec1 = df[[f'x_{col2[2:]}', f'y_{col2[2:]}', f'z_{col2[2:]}']].values - df[[f'x_{col1[2:]}', f'y_{col1[2:]}', f'z_{col1[2:]}']].values
        vec2 = df[[f'x_{col3[2:]}', f'y_{col3[2:]}', f'z_{col3[2:]}']].values - df[[f'x_{col2[2:]}', f'y_{col2[2:]}', f'z_{col2[2:]}']].values
        
        
        # Compute dot product and magnitudes
        dot_product = np.sum(vec1 * vec2, axis=1)
        magnitude1 = np.linalg.norm(vec1, axis=1)
        magnitude2 = np.linalg.norm(vec2, axis=1)
        
        # Avoid division by zero or very small values
        mask = (magnitude1 * magnitude2) != 0
        dot_product[mask] /= (magnitude1[mask] * magnitude2[mask])
        
        # Clip values to prevent invalid input to arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Compute angle between vectors
        angles[angle_col] = np.arccos(dot_product) * (180 / np.pi)
    
    return pd.DataFrame(angles)

def calculate_hand_motion_features(df, landmark_cols):
    """
    Feature engineers these new features: 
        - normalized landmarks locations  ✅
        - velocity (temporal feature) ✅
        - acceleration (temporal feature) ✅
        - relative pairwise distances (Euclidean distance) ✅
        - relative landmark angles
    """

    new_cols = {}

    for col in landmark_cols:
        new_cols[f"velocity_{col}"] = df[col].diff().fillna(0)
        new_cols[f"acceleration_{col}"] = new_cols[f"velocity_{col}"].diff().fillna(0)

    # Calculate pairwise distancces between all landmarks 
    landmark_pairs = list(combinations(landmark_cols, 2))
    for (col1, col2) in landmark_pairs:
        idx1 = col1[1:]
        idx2 = col2[1:]

        if idx1 == idx2:
            continue
        x1, y1, z1 = f'x{idx1}', f'y{idx1}', f'z{idx1}'
        x2, y2, z2 = f'x{idx2}', f'y{idx2}', f'z{idx2}'
        distance_col = f'distance_{idx1}_{idx2}'
        new_cols[distance_col] = np.sqrt((df[x1] - df[x2])**2 + (df[y1] - df[y2])**2 + (df[z1] - df[z2])**2)
    
    # Normalized Landmark locations
    normalized_df = normalize_landmark_locations(df.copy(), landmark_cols)
    new_cols.update(normalized_df.filter(regex="norm").to_dict())

    # # Relative Landmark angles
    angle_df  = calculate_landmark_angles(df, landmark_cols)
    new_cols.update(angle_df.to_dict())

    new_df = pd.DataFrame(new_cols)
    return pd.concat([df, new_df], axis=1)


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
        return pd.concat(data_frames, ignore_index=True).dropna(), landmark_cols, landmark_world_cols
    
def reshape_for_lstm(X, time_steps=30):
    n_samples = len(X) // time_steps
    X = X.iloc[:n_samples * time_steps, :]  # Ensure even division
    return X.values.reshape(n_samples, time_steps, -1)

def main():
    input_dir = "data/"
    dataframe, landmark_cols, landmark_world_cols = create_dataframe_from_data(input_dir)

    dataframe.to_csv('model/LSTM/df_combined.csv')

    X = dataframe.drop(columns=['gesture'], axis=1)
    y = dataframe['gesture']
    
    n_splits = 5
    split_ratio = 0.2
    for train_index, test_index in TimeSeriesSplit(n_splits=n_splits).split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Split the remaining train data into train and validation
        val_index = int(len(X_train) * split_ratio)
        X_val, X_train = X_train.iloc[:val_index], X_train.iloc[val_index:]
        y_val, y_train = y_train.iloc[:val_index], y_train.iloc[val_index:]
        
        # Apply feature engineering to X_train/X_val/X_test
        # X_train = calculate_hand_motion_features(X_train, landmark_cols)
        # X_val = calculate_hand_motion_features(X_val, landmark_cols)
        # X_test = calculate_hand_motion_features(X_test, landmark_cols)

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
            # ('pca', PCA(n_components=2))
        ])

        X_train_transformed = numerical_transformer.fit_transform(X_train)
        X_val_transformed = numerical_transformer.transform(X_val)
        X_test_transformed = numerical_transformer.transform(X_test)
        
        X_train_lstm = reshape_for_lstm(pd.DataFrame(X_train_transformed))
        X_val_lstm = reshape_for_lstm(pd.DataFrame(X_val_transformed))
        X_test_lstm = reshape_for_lstm(pd.DataFrame(X_test_transformed))
        
        y_train_lstm = y_train.values[:X_train_lstm.shape[0] * X_train_lstm.shape[1]].reshape(-1, 1)
        y_val_lstm = y_val.values[:X_val_lstm.shape[0] * X_val_lstm.shape[1]].reshape(-1, 1)
        y_test_lstm = y_test.values[:X_test_lstm.shape[0] * X_test_lstm.shape[1]].reshape(-1, 1)
        
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_lstm, y_train_lstm, epochs=10, validation_data=(X_val_lstm, y_val_lstm))
        
        test_loss, test_acc = model.evaluate(X_test_lstm, y_test_lstm)
        print(f'Test accuracy: {test_acc}')

   



main()