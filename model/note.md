we need to normalize the raw xyz coords


feature engineering - transforming raw data and creating features to improve the prediction 

feature selection - selecting key dataset features to reduce dimensionality


    from sklearn.model_selection import TimeSeriesSplit

    def main():
        input_dir = "data/"
        dataframe = create_dataframe_from_data(input_dir).dropna()

        X = dataframe.drop(columns=['gesture'], axis=1)
        y = dataframe['gesture']

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Perform feature engineering on X_train and X_test...

        # Use TimeSeriesSplit for cross-validation on the training set
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Train and evaluate your model on each fold...



    """
    method 1 in spliting the data
    for a given gesture_index, take 70 for trainning and 15/15 for val and test
    """


The quote "Feature engineering should focus on extracting features from the entire sequence rather than individually" is crucial for time series data like gestures.

Your current implementation seems to address this by calculating features like velocity and acceleration, which depend on multiple data points within the sequence (not individual frames).
However, the relative landmark angles might need further consideration. Are you calculating angles for each frame within the sequence, or for the entire gesture?
Recommendations:

If you intend to capture the overall gesture motion through angles, consider calculating them over the entire sequence (e.g., average or maximum angle) instead of for each frame.
Explore additional features that capture the gesture dynamics across the entire sequence, such as range of motion for each landmark or average speed.
By implementing these solutions and refining your feature engineering based on the "important quote," you can ensure consistent data shapes and potentially improve your gesture recognition model's performance.