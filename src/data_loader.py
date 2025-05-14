import pandas as pd
from pathlib import Path

def load_cancer_data(data_path: str) -> pd.DataFrame:
    """
    Parameters:
    - data_path (str): Path to the dataset CSV file
    Returns:
    - pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(data_path)
        print(f"[INFO] Dataset loaded successfully with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found at {data_path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        raise

def show_basic_info(df: pd.DataFrame, n: int = 5) -> None:
    """
    Parameters:
    - df (pd.DataFrame): The dataset
    - n (int): Number of rows to preview
    """
    print("\n[INFO] First rows of the dataset:")
    print(df.head(n))
    print("\n[INFO] Dataset info:")
    print(df.info())
    print("\n[INFO] Summary statistics:")
    print(df.describe(include='all'))
