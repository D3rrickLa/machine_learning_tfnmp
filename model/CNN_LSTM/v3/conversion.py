# juypter notebook keeps freezing VS when running the compression, will do it here

import gc
import os 
import time 
import numpy as np 
import pandas as pd 

from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, Callback
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.metrics import Precision, Recall, F1Score
from keras._tf_keras.keras.optimizers import Adam , RMSprop, Nadam, SGD
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, InputLayer, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D, LayerNormalization, Activation, GRU, Attention
from keras._tf_keras.keras.regularizers import L1L2, L1, L2
from keras._tf_keras.keras.utils import to_categorical
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from model.CNN_LSTM.components.engineering import FeatureEngineering
from model.CNN_LSTM.components.dfmodify import DataframeModify 
from model.CNN_LSTM.components.dfcreation import DataframeCreate, DataframeSave
from model.CNN_LSTM.components.custom_keras_callbacks import CustomEarlyStopping

def preprocess_pipeline(timeseries_columns: list, numerical_columns: list, categorical_columns: list = None):
    ts_numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)), # might want to change this out back to the interpolatioon methods
        ('imputer2', SimpleImputer(strategy="mean")),
        ('scaler', MinMaxScaler()) # we should use minmax?
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ("normalize", MinMaxScaler())
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
        n_jobs=1
    )
 
    preprocessor.set_output(transform="pandas")
    
    return preprocessor

def create_graphs(history_dev_1, cm, class_labels):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set Evaluation')
    plt.ion()
    plt.show()

    # Extracting the history
    train_loss = history_dev_1.history['loss']
    val_loss = history_dev_1.history['val_loss']
    train_acc = history_dev_1.history['accuracy']
    val_acc = history_dev_1.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    # Create a figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training and validation loss
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:blue')
    ax1.plot(epochs, val_loss, label='Validation Loss', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot training and validation accuracy
    ax2.plot(epochs, train_acc, label='Train Accuracy', color='tab:green')
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='tab:red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Display the plots
    plt.show()

input_path = r"C:\Users\Gen3r\Documents\capstone\ml_model\data\data_3"
dataframe = DataframeCreate.create_dataframe_from_data(input_path=input_path)
X_train, y_train, X_val, y_val, X_test, y_test = DataframeCreate.split_dataset(df=dataframe, target_label='gesture', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

landmark_columns = [f"{col}" for col in dataframe.columns if col.startswith(("hx", "hy", "hz", "px", "py", "pz", "lx", "ly", "lz", "rx", "ry", "rz"))]
categorical_columns = ["gesture_index"]
numerical_columns = ["frame", "frame_rate", "frame_width", "frame_height"] + [f"{col}" for col in dataframe.columns if col.startswith("pose_visibility")]
# derived_features =  [f"{feat}_{col}" for feat in ["velocity", "acceleration", "jerk"] for col in landmark_columns if col.startswith(("lx", "ly", "lz", "rx", "ry", "rz"))]
time_series_columns = landmark_columns  # + derived_features   
# res = [item for item in landmark_columns if item.startswith(("r", "l"))]

# augmenting train data
X_train_augmented_1, y_train_augmented_1 = DataframeModify.augment_model(X_train, y_train, noise_level=0.05, translation_vector=[0.6, -0.5, 0.005], rotation_angle=45)
X_train_augmented_2, y_train_augmented_2 = DataframeModify.augment_model(X_train, y_train, noise_level=0.04, translation_vector=[0.2, 0.76, -0.15], rotation_angle=15)
X_train_augmented_3, y_train_augmented_3 = DataframeModify.augment_model(X_train, y_train, noise_level=0.023, translation_vector=[-0.4, 0.3, 0.1], rotation_angle=30)

X_train_precombined = pd.concat([X_train, X_train_augmented_1, X_train_augmented_2, X_train_augmented_3], axis=0, ignore_index=True)
y_train_precombined = pd.concat([y_train, y_train_augmented_1, y_train_augmented_2, y_train_augmented_3], axis=0, ignore_index=True)

gc.collect()

X_train_augmented_4, y_train_augmented_4 = DataframeModify.augment_model(X_train_precombined, y_train_precombined, noise_level=0.01, translation_vector=[-0.21, -0.3, -0.01], rotation_angle=-23)

X_train_combined = pd.concat([X_train_precombined, X_train_augmented_4], axis=0, ignore_index=True)
y_train_combined = pd.concat([y_train_precombined, y_train_augmented_4], axis=0, ignore_index=True)

gc.collect()
print("finish augmentation")

X_train_transformed, X_val_transformed, X_test_transformed = None, None, None
Flag = False

if os.path.exists("X_train_transformed.csv.gz") and Flag == True:
    print("loaded from drive")
    X_train_transformed = pd.read_csv("X_train_transformed.csv.gz", compression="gzip")
    X_val_transformed = pd.read_csv("X_val_transformed.csv.gz", compression="gzip")
    X_test_transformed = pd.read_csv("X_test_transformed.csv.gz", compression="gzip")
else:
    print("starting new calculations")
    preprocessor = preprocess_pipeline(time_series_columns, numerical_columns, categorical_columns)
    X_train_transformed = preprocessor.fit_transform(X_train_combined)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    DataframeSave.save_dataframe(X_train_transformed, X_val_transformed, X_test_transformed)
