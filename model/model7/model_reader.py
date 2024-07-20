import numpy as np
import pandas as pd

def load_from_numpy(file_path):
    # Load the structured array from the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Convert the structured array to a pandas DataFrame
    df = pd.DataFrame(data)
    
    return df

# Example usage:
file_path = "data/data_3/EAT_1721432096758958300.npy"  # Replace with your actual file path
df = load_from_numpy(file_path)
print(df.head())