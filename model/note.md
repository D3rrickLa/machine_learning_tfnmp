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