





import os
import numpy as np
import pandas as pd


def create_dataframe_from_data(input_path: str) -> pd.DataFrame:
    dataframes = []
    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)

        data = np.load(file_path, file_name)