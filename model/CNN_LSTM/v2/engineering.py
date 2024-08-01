import numpy as np
import pandas as pd


class feature_engineering():
    def calculate_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:
        elapsed_lists = []

        for _, gesture_data in df.groupby("gesture_index"):
            avg_frame_rate = np.mean(gesture_data["frame_rate"])

            for i in gesture_data["frame"]:
                elapsed_lists.append(i / avg_frame_rate)
            
        df['elapsed_time'] = elapsed_lists

        return df

    def calculate_temporal_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
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
    
    def calculate_temporal_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
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

    def calculate_landmark_distances(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        distance_columns = [f"lm_distance_{i}_{j}" for i in range(len(cols)//3) for j in range(len(cols)//3)]

        for _, gesture_data in df.groupby("gesture_index"):
            gesture_data = gesture_data.sort_values(by="frame")
            
            coords = gesture_data[cols].values.reshape(-1, len(cols) // 3, 3)
            distances = np.sqrt(np.sum((coords[:, :, None] - coords[:, None, :])**2, axis=-1))
            
            # we technically should do something called zero out - basically in the df x_0/x_1 == x_1/x_0 (redundant)

            distances_flat = distances.reshape(-1, len(distance_columns))
            df.loc[gesture_data.index, distance_columns] = distances_flat

        return df

    def calculate_landmark_angles(df: pd.DataFrame, cols: list) -> pd.DataFrame:
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