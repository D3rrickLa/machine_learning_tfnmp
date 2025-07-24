import math
import os
import time
from typing import Tuple
import numpy as np
import pandas as pd

class DataframeModify():
    def augment_model(df: pd.DataFrame, y_df: pd.DataFrame, noise_level=0.0, translation_vector=None, rotation_angle=0.0, duplicate_y=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_augmented = df.copy()

        landmark_columns = [f"{col}" for col in df_augmented.columns if col.startswith(("hx", "hy", "hz", "px", "py", "pz", "lx", "ly", "lz", "rx", "ry", "rz"))]
        num_body_parts = ("h", "p", "l", "r")

        x_columns = [col for col in landmark_columns if any(col.startswith(f'{i}x') for i in num_body_parts)] # this way works because of how i is defined before hand... don't really know
        y_columns = [col for col in landmark_columns if any(col.startswith(f'{i}y') for i in num_body_parts)]
        z_columns = [col for col in landmark_columns if any(col.startswith(f'{i}z') for i in num_body_parts)]

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, df[x_columns + y_columns + z_columns].shape)
            df_augmented[x_columns + y_columns + z_columns] += noise

        # Apply translation
        if translation_vector is not None:
            for i, col in enumerate(x_columns):
                df_augmented[col] += translation_vector[i % 3]
            for i, col in enumerate(y_columns):
                df_augmented[col] += translation_vector[i % 3]
            for i, col in enumerate(z_columns):
                df_augmented[col] += translation_vector[i % 3]

        # Apply rotation around the Z-axis
        if rotation_angle != 0:
            angle_radians = np.radians(rotation_angle)
            cos_angle = np.cos(angle_radians)
            sin_angle = np.sin(angle_radians)

            for col in x_columns:
                y_col = col.replace('x', 'y')
                df_augmented[col], df_augmented[y_col] = (cos_angle * df_augmented[col] - sin_angle * df_augmented[y_col],
                                                        sin_angle * df_augmented[col] + cos_angle * df_augmented[y_col])
        
        # making the gesture index of the augment different - will be added back to the the df
        # will need to double up on the y train and test as well - in reshape_y_labels
        if "gesture_index" in df_augmented.columns:
            cur_time = time.time_ns()
            df_augmented["gesture_index"] += cur_time
        
        y_df_combined = None
        if duplicate_y:
            y_df_combined = pd.concat([y_df, y_df]).reset_index(drop=True)
        else:
            print("duplicated y is set to False")
            y_df_combined = y_df.reset_index(drop=True)

        return df_augmented, y_df_combined

    @staticmethod
    def transform_to_sequenceses(df: pd.DataFrame, sequence_length, target: str, additional_targets: list = None) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        labels = []
        grouped = df.groupby('remainder__gesture_index')
        
        for _, group in grouped:
            group = group.sort_values('num__frame').reset_index(drop=True)
            for i in range(0, len(group) - sequence_length + 1, sequence_length):
                sequence = group.iloc[i:i+sequence_length].drop(columns=['remainder__gesture_index', target] + (additional_targets if additional_targets else [])).values
                if len(sequence) == sequence_length:
                    sequences.append(sequence)
                    labels.append(group.iloc[0][target]) if additional_targets is None else labels.append(group.iloc[0][[target] + additional_targets])
        return np.array(sequences), np.array(labels)

    # Assuming the target column name is 'gesture' and no additional targets
    def create_sequences_with_labels(X_transformed, y, sequence_length) -> Tuple[np.ndarray, np.ndarray]:
        # Combine features and labels into a single DataFrame
        combined_df = pd.concat([pd.DataFrame(X_transformed), y.reset_index(drop=True)], axis=1)
        
        # Convert the DataFrame to sequences
        X_sequences, y_sequences = DataframeModify.transform_to_sequenceses(combined_df, sequence_length, target='gesture')
        
        return X_sequences, y_sequences