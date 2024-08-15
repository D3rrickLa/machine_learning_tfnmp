import math
import os
from typing import Tuple
import numpy as np
import pandas as pd

class DataframeCreate():
    def create_dataframe_from_data(input_path: str) -> pd.DataFrame:
        dataframes = []
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)

            data = np.load(file_path, allow_pickle=True)
            df = pd.DataFrame(data)

            gesture = file_name.split("_")[0]
            gesture_index = int(file_name.split("_")[1].split(".")[0]) 

            # Adds 2 new columns to df 
            df["gesture"] = gesture 
            df["gesture_index"] = gesture_index 

            dataframes.append(df)

        return pd.concat(dataframes, ignore_index=True) if len(dataframes) > 0 else ValueError("Dataframe is empty")
    
    @staticmethod
    def create_dict_from_df(df: pd.DataFrame) -> dict:
        diction = {x: [] for x in np.unique(df["gesture"].values.tolist())}
        for gesture_index, gesture_data in df.groupby("gesture_index"):
            gesture = np.unique(gesture_data["gesture"].values.tolist())[0]
            tmp = diction[gesture] + [gesture_index]
            diction.update({gesture:tmp})
        return diction
    
    def split_dataset(df: pd.DataFrame, target_label: str, additional_targets: list=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert train_ratio + val_ratio + test_ratio == 1.0, "ratios must sum to 1."

        gesture_index_dict = DataframeCreate.create_dict_from_df(df)

        train_indices, val_indices, test_indices = [], [], []

        for _, indices in gesture_index_dict.items():
            n_total = len(indices)
            n_train, n_val = math.ceil(n_total * train_ratio), math.ceil(n_total * test_ratio)

            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train+n_val])
            test_indices.extend(indices[n_train+n_val:])
        
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

        def split_data(data: pd.DataFrame, target_label: str, additional_targets=None):
            X = data.drop(columns=[target_label])
            y = data[[target_label] + additional_targets] if additional_targets else data[[target_label]]
            return X, y

        X_train, y_train = split_data(train_set, target_label, additional_targets)
        X_val, y_val = split_data(val_set, target_label, additional_targets)
        X_test, y_test = split_data(test_set, target_label, additional_targets)

        return X_train, y_train, X_val, y_val, X_test, y_test

class DataframeSave():
    def save_dataframe(X_train, X_val, X_test) -> None:
        pd.DataFrame.to_csv(X_train, "X_train_transformed.csv.gz", index=False, compression="gzip")
        pd.DataFrame.to_csv(X_val, "X_val_transformed.csv.gz", index=False, compression="gzip")
        pd.DataFrame.to_csv(X_test, "X_test_transformed.csv.gz", index=False, compression="gzip")