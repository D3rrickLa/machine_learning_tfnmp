import math
import os  
import time 

from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras._tf_keras.keras.metrics import MeanAbsoluteError, Accuracy, Precision, Recall, MeanSquaredError
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import Adam , RMSprop, Nadam
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Masking, InputLayer, Conv1D, MaxPooling1D, Flatten, TimeDistributed, LayerNormalization, Activation
from keras._tf_keras.keras.regularizers import L1L2, L1, L2
from keras._tf_keras.keras.utils import to_categorical
import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder 

from FeatureEngineering import FeatureEngineering as fe 

def create_dataframe_from_data(input_path: str):
    dataframes = []
    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)

        data = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(data)

        gesture = file_name.split("_")[0]
        gesture_index = int(file_name.split("_")[1].split(".")[0])
        
        # add two new columns to df 
        df["gesture"] = gesture
        df["gesture_index"] = gesture_index

        dataframes.append(df)
        df.sort_values(by="frame", inplace=True)
    
    return pd.concat(dataframes, ignore_index=True) if len(dataframes) > 0 else ValueError("Dataframe is empty")

def create_dict_from_df(df: pd.DataFrame):
    diction = {x: [] for x in np.unique(df["gesture"].values.tolist())}
    for gesture_index, gesture_data in df.groupby("gesture_index"):
        gesture = np.unique(gesture_data["gesture"].values.tolist())[0]
        tmp = diction[gesture] + [gesture_index]
        diction.update({gesture:tmp})
    return diction

def split_dataset(df: pd.DataFrame, target_label: str, additional_targets: list=None, train_ratio=0.8, test_ratio=0.2):
    assert train_ratio + test_ratio == 1.0, "ratios must sum to 1."

    gesture_index_dict = create_dict_from_df(df)

    train_indices = []
    test_indices = []

    for _, indices in gesture_index_dict.items():
        n_train = math.ceil(len(indices) * train_ratio)
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])

    grouped_data = df.groupby("gesture_index")

    train_frames, test_frames = [], []
    for idx in train_indices:
        train_frames.append(grouped_data.get_group(idx))
        
    for idx in test_indices:
        test_frames.append(grouped_data.get_group(idx))

    # Concatenate the dataframes to create the final train and test sets
    train_set = pd.concat(train_frames).reset_index(drop=True)
    test_set = pd.concat(test_frames).reset_index(drop=True)

    # Separate X and y
    X_train = train_set.drop(columns=[target_label])
    y_train = train_set[[target_label] + additional_targets] if additional_targets else train_set[[target_label]]
    X_test = test_set.drop(columns=[target_label])
    y_test = test_set[[target_label] + additional_targets] if additional_targets else test_set[[target_label]]

    return X_train, y_train, X_test, y_test

def augment_model(df: pd.DataFrame, noise_level=0.0, translation_vector=None, rotation_angle=0.0):
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
    # will need to double up on the y train and test as well
    if "gesture_index" in df_augmented.columns:
        cur_time = time.time_ns()
        df_augmented["gesture_index"] += cur_time

    return df_augmented

def calculate_hand_motion_feature(df: pd.DataFrame, landmark_cols: list):
    df_copy = df.copy()
    print(df_copy.shape)
    s = time.process_time()
    df_elapsed = fe.calculate_elapsed_time(df_copy)    
    print(time.process_time()-s)

    s = time.process_time()
    df_temporal = fe.calculate_temporal_features(df_copy, landmark_cols)
    print(time.process_time()-s)
    # df_stats = fe.calculate_temporal_stats(df_copy, landmark_cols)
    # df_pairwise = fe.calculate_landmark_distances(df_copy, landmark_cols)
    # df_angle = fe.calculate_landmark_angles(df_copy, landmark_cols)
    # df_combined = pd.concat([df_copy, df_angle], axis=1)
    
    # Ensure there are no duplicate columns
    df_combined = df_copy.loc[:,~df_copy.columns.duplicated()]
    return df_combined

def preprocess_pipeline(timeseries_columns, numerical_columns, categorical_columns):
    ts_numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)), # might want to change this out back to the interpolatioon methods
        ('imputer2', SimpleImputer(strategy="mean")),
        ('scaler', MinMaxScaler())
        # ('smoother', FunctionTransformer(lambda x: x.rolling(window=3, min_periods=1).mean())),
        # ('differencing', FunctionTransformer(lambda x: x.diff().fillna(0))),
        # ('lag_features', FunctionTransformer(lambda x: pd.concat([x.shift(i) for i in range(1, 4)], axis=1).fillna(0))),
        # ('rolling_stats', FunctionTransformer(lambda x: pd.concat([x.rolling(window=3).mean(), x.rolling(window=3).std()], axis=1).fillna(0)))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        # ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Polynomial features
        # ('power', PowerTransformer(method='yeo-johnson')),   
        ("normalize", StandardScaler()),
        ('scaler', MinMaxScaler()),
        ('pca', PCA(n_components=10))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")), # technically this is wrong
        ("ohe", OneHotEncoder(sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ts_num', ts_numerical_transformer, timeseries_columns),
            ('num', numerical_transformer, numerical_columns),
            # ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='passthrough',
        sparse_threshold=0,
        n_jobs=-1
    )
 
    preprocessor.set_output(transform="pandas")
    
    return preprocessor

def reshape_y_labels(df : pd.DataFrame):
    """
    reasoning for this function:
    a single gesture recording can have the same index of n frames.
    This will balloon the size of the (x, y), when in reality it's much
    smaller. This will goo through each frame, and if the continuous pattern
    breaks (e.g. 12 -> 0), everything before "12" will be removed keeping only
    one instance of the gesture
    """
    unique_sequences = []
    for _, group in df.groupby("gesture_index"):
        reset_points = group['frame'].diff().fillna(1) < 0
        if reset_points.any():
            unique_sequences.append(group[reset_points])
        else:
            # If no reset points, consider the whole group as unique
            unique_sequences.append(group.iloc[[0]])

    # Concatenate unique sequences
    df_unique = pd.concat(unique_sequences).reset_index(drop=True)
    return pd.factorize(df_unique["gesture"])

def create_lstm(input_shape, output_units):
    model = Sequential()
    model.add(InputLayer(shape=(input_shape)))
    model.add(Masking(mask_value=-1))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(np.unique(output_units)), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

input_path = "data/data_3"
dataframe = create_dataframe_from_data(input_path)
X_train, y_train, X_test, y_test = split_dataset(dataframe, "gesture", ["frame", "gesture_index"])

landmark_columns = [f"{col}" for col in dataframe.columns if col.startswith(("hx", "hy", "hz", "px", "py", "pz", "lx", "ly", "lz", "rx", "ry", "rz"))]
categorical_columns = ["gesture_index"]
numerical_columns = ["frame", "frame_rate", "frame_width", "frame_height"] + [f"{col}" for col in dataframe.columns if col.startswith("pose_visibility")]
derived_features =  ['elapsed_time'] + \
                    [f"{feat}_{col}" for feat in ["velocity", "acceleration", "jerk"] for col in landmark_columns if col.startswith(("lx", "ly", "lz", "rx", "ry", "rz"))]
time_series_columns = landmark_columns + derived_features     
res = [item for item in landmark_columns if item.startswith(("r", "l"))]

if not os.path.exists("model/model8/train.csv"):
    X_train_augmented = augment_model(X_train, noise_level=0.05, translation_vector=[0.6, -0.5, 0.05], rotation_angle=45)
    X_test_augmented = augment_model(X_test, noise_level=0.05, translation_vector=[0.6, -0.5, 0.05], rotation_angle=45)

    X_train_combined = pd.concat([X_train, X_train_augmented], axis=0, ignore_index=True)
    X_test_combined = pd.concat([X_test, X_test_augmented], axis=0, ignore_index=True)
    
    X_train_fe = calculate_hand_motion_feature(X_train_combined, res)
    X_test_fe = calculate_hand_motion_feature(X_test_combined, res)

    X_train_fe.to_csv("model/model8/train.csv", index=False)
    X_test_fe.to_csv("model/model8/test.csv", index=False)
else:
    X_train_fe = pd.read_csv("model/model8/train.csv")
    X_test_fe = pd.read_csv("model/model8/test.csv")
    print("imported")

preprocessor = preprocess_pipeline(time_series_columns, numerical_columns, categorical_columns)
X_train_transformed = preprocessor.fit_transform(X_train_fe)
X_test_transformed = preprocessor.transform(X_test_fe) # need to redo that gesture_index thing

y_train_reshaped, labels = reshape_y_labels(y_train)
y_test_reshaped, _ = reshape_y_labels(y_test)
class_labels = labels

y_train_one_hot = to_categorical(y_train_reshaped, num_classes=len(labels))
y_test_one_hot = to_categorical(y_test_reshaped, num_classes=len(labels))

print(X_train_fe.shape, X_test_fe.shape)
X_train_reshaped = np.reshape(X_train_fe, ((X_train_fe.shape[0]//30), 30, X_train_fe.shape[1]))
X_test_reshaped = np.reshape(X_test_fe, ((X_test_fe.shape[0]//30), 30, X_train_fe.shape[1]))

print("before lstm:",X_train_reshaped.shape, X_test_reshaped.shape, y_train_one_hot.shape, y_test_one_hot.shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)	
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-10)	
model_checkpoint = ModelCheckpoint(filepath=f"checkpoints/checkpoint{time.time()}.model.keras", mode="max", monitor="val_accuracy", save_best_only=True)

input =  (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
model = create_lstm(input, labels)
history = model.fit(
    X_train_reshaped, y_train_one_hot, 
    epochs=200,  
    validation_split=0.20,
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

test_loss, test_acc = model.evaluate(X_test_reshaped, y_test_one_hot)	
print(f'Test Accuracy: {test_acc} || Test Loss: {test_loss}')