import cv2 
import mediapipe as mp 
import numpy as np 
import time 
import pandas as pd 
import os 
import os 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from keras._tf_keras.keras.utils import to_categorical 
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.layers import LSTM, Dense
from keras._tf_keras.keras.models import Sequential

# Define functions to calculate velocity and acceleration
def calculate_velocity(data):
    velocity = np.diff(data, axis=0)
    return np.vstack([velocity, np.zeros((1, velocity.shape[1]))])

def calculate_acceleration(data):
    acceleration = np.diff(data, n=2, axis=0)
    return np.vstack([acceleration, np.zeros((2, acceleration.shape[1]))])

input_path = "model/model5/data/dataset4"

data_frames = [] 
gesture_index = 0 

# creates a dataframe containing information from all the imported CSV files
for file_name in os.listdir(input_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_path, file_name)
        df = pd.read_csv(file_path)

        # extracts the gesture name and index from file name
        gesture_action = file_name.split("_")[0] # 0 or 1 - left or right of split
        gesture_index = int(file_name.split("_")[1].split(".")[0])


        # adds gesture name and index as a column into the DataFrame
        df['gesture'] = gesture_action
        df['gesture_index'] = gesture_index

        df.sort_values(by="frame", inplace=True)
        data_frames.append(df)


concat_data = pd.concat(data_frames, ignore_index=True)



excluded_cols = ['frame', 'gesture', 'gesture_index', 'frame_rate', 'frame_width', 'frame_height']

# landmark coordinates (exclude everything that isn't X, Y, and or Z)
landmark_cols = [col for col in concat_data.columns if col not in excluded_cols]

# Preprocessing --------------

# Normalize and Standardize the features
scaler = StandardScaler()
concat_data[landmark_cols] = scaler.fit_transform(concat_data[landmark_cols])


# Feature Engineering/Extraction -----------

# Calculate distances between specific points, 0 and 1 in this case
concat_data['distance_0_1'] = ((concat_data['x_0'] - concat_data['x_1'])**2 + 
                            (concat_data['y_0'] - concat_data['y_1'])**2 + 
                            (concat_data['z_0'] - concat_data['z_1'])**2) ** 0.5


# Calculate the acceleration and velocity between points
velocity_cols = []
acceleration_cols = []

for i in range(0, 21):  # 21 landmark points
    for axis in ['x', 'y', 'z']:
        col_name = f'{axis}_{i}'
        data = concat_data[col_name].values.reshape(-1, 1)
        
        velocity = calculate_velocity(data)
        acceleration = calculate_acceleration(data)
        
        # Store velocity and acceleration in a temporary DataFrame
        velocity_col_name = f'velocity_{col_name}'
        acceleration_col_name = f'acceleration_{col_name}'
        
        velocity_df = pd.DataFrame(velocity, columns=[velocity_col_name])
        acceleration_df = pd.DataFrame(acceleration, columns=[acceleration_col_name])
        
        velocity_cols.append(velocity_df)
        acceleration_cols.append(acceleration_df)

# Concatenate all columns at once to avoid DataFrame fragmentation
velocity_data = pd.concat(velocity_cols, axis=1)
acceleration_data = pd.concat(acceleration_cols, axis=1)

# Combine original data with velocity and acceleration data
concat_data = pd.concat([concat_data, velocity_data, acceleration_data], axis=1)

combined_cols = landmark_cols + ['distance_0_1'] + velocity_data.columns.to_list() + acceleration_data.columns.tolist()

# Segmenting data - gesture can vary ing length, this will segment then into fixed-length windows
max_timesteps = 45 
sequences = [group[combined_cols].values for _, group in concat_data.groupby('gesture')]
padded_sequences = pad_sequences(sequences, maxlen=max_timesteps, padding='post', truncating='post')


features = [col for col in concat_data.columns if col not in ["frame", "gesture"]]

X = concat_data[features].values

ladel_encoder = LabelEncoder()
y = ladel_encoder.fit_transform(concat_data['gesture'])

one_hot_labels = to_categorical(y)



# Reshape X for LSTM input (samples, timesteps, features)
X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)



model = Sequential([
    LSTM(128, input_shape=(1, len(features))),
    Dense(len(one_hot_labels), activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f'Test Accuracy: {test_acc}')