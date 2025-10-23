import pandas as pd
import numpy as np

def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the situation of missing values in the dataset
    """
    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.isnull().count() * 100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    
    return np.transpose(tt)

def analyze_frequent_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the most frequent values in the dataset
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    items = []
    vals = []
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(f"处理列 {col} 时出错: {ex}")
            items.append(0)
            vals.append(0)
            continue
    
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    
    return np.transpose(tt)

def analyze_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the number of unique values in the dataset
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    
    tt['Uniques'] = uniques
    return np.transpose(tt)