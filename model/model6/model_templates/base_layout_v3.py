# the 3rd version of the base layout
# 2nd one had coding problem, looks good for a template
# but can't do LSTM
from functools import partial
from itertools import combinations
import os


from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

input_dir = "model/model6/data/dataset5/"


def calcualte_hand_motion_features(df, landmark_cols):
    """
    function to feature engineer 3 new features:
    velocity, accleration, and pairwise distances between all ladmarks 

    returns a pandas dataframe
    """
    new_cols = {}

    for col in landmark_cols:
        new_cols[f"velocity_{col}"] = df[col].diff().fillna(0)

        new_cols[f"acceleration_{col}"] = new_cols[f"velocity_{col}"].diff().fillna(0)
        
    # Calculate pairwise distances between all landmarks
    landmark_pairs = list(combinations(landmark_cols, 2))
    for (col1, col2) in landmark_pairs:
        idx1 = col1[1:]  # Get index part from 'x0', 'y0', etc.
        idx2 = col2[1:]
        if idx1 == idx2:
            continue
        x1, y1, z1 = f'x{idx1}', f'y{idx1}', f'z{idx1}'
        x2, y2, z2 = f'x{idx2}', f'y{idx2}', f'z{idx2}'
        distance_col = f'distance_{idx1}_{idx2}'
        new_cols[distance_col] = np.sqrt((df[x1] - df[x2])**2 + (df[y1] - df[y2])**2 + (df[z1] - df[z2])**2)
    
    new_df = pd.DataFrame(new_cols)

    return pd.concat([df, new_df], axis=1)


def create_dataframe_from_dataset(input_path):
    """
    How the model works:
    data coming in are multiple CSV. Each CSV
    represents 1 hand gesture motion. There are multiple hand gestures
    inside that input path (e.g. SLEEPING_1-N, SWIPE_LEFT_1-N, etc.)

    The funciton will combine all those CSV into 1 dataframe and 
    create 2 new coloumns: gesture and gesture_index
    """

    data_frames = []
    landmark_cols = []
    gesture_index = 0 

    for file_name in os.listdir(input_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_path, file_name)

            dataframe = pd.read_csv(file_path)

            if(len(landmark_cols) == 0):
                landmark_cols = [col for col in dataframe.columns if col.startswith(("x", "y", "z"))]
            
            gesture = file_name.split("_")[0]
            gesture_index = int(file_name.split("_")[1].split(".")[0])

            dataframe["gesture"] = gesture
            dataframe["gesture_index"] = gesture_index

            dataframe.sort_values(by="frame", inplace=True)

            # this is where we will do the feature extraction, simpler to do it 
            # with the data coming in rather than on the whole set

            dataframe = calcualte_hand_motion_features(dataframe.copy(), landmark_cols=landmark_cols)

            data_frames.append(dataframe)
    
    if len(data_frames) == 0:
        raise ValueError("Dataframe has not data")
    else:
        return pd.concat(data_frames, ignore_index=True)
    

data = create_dataframe_from_dataset(input_dir).dropna()


# seperate the features (X) and targets (y)
X = data.drop(columns=['gesture'], axis=1)
y, class_labels = pd.factorize(data['gesture'])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# part of reshaping to match LSTM
def create_sequences(data, labels, timesteps):
    sequences = []
    sequence_labels = []
    for i in range(len(data)- timesteps):
        sequences.append(data[i:i + timesteps])
        sequence_labels.append(labels[i + timesteps])

    return np.array(sequences), np.array(sequence_labels)

# Reshaping X to sequences for LSTM
timesteps = 10 # length of sequences 
num_features = X_train_scaled.shape[1]

X_train_reshaped, y_train_reshaped = create_sequences(X_train_scaled, y_train, timesteps)
X_val_reshaped, y_val_reshaped = create_sequences(X_val_scaled, y_val, timesteps)
X_test_reshaped, y_test_reshaped = create_sequences(X_train_scaled, y_train, timesteps)


num_features = X_train_reshaped.shape[2]
model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, num_features)))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# need to change this to an actual val set
model.fit(X_test_reshaped, y_train_reshaped, epochs=20, batch_size=32, validation_data=(X_val_reshaped, y_val_reshaped))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test_reshaped)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predictions
y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)

model.save("model/model6/data/lstm_v2.keras")

# get the class labels
class_labels_df = pd.DataFrame({'gesture': class_labels})
class_labels_df.to_csv("model/model6/data/class_labels.csv", index=False)