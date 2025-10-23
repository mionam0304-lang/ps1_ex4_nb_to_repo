import pandas as pd
from pathlib import Path

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    loading train_data and test_data
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def combine_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    combining train_data and test_data
    """
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df

def get_data_info(df: pd.DataFrame) -> None:
    """
    print the basic information of data
    """
    print("data shape:", df.shape)
    print("\nfirst 5 lines:")
    print(df.head())
    print("\ndata type:")
    print(df.info())
    print("\ndescriptive statistics:")
    print(df.describe())