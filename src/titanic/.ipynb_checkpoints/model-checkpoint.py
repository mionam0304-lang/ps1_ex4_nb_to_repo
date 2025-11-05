import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def prepare_features(df: pd.DataFrame, predictors: list, target: str) -> tuple:
    """
    Prepare the feature data and labels
    """
    X = df[predictors]
    y = df[target].values
    return X, y

def train_model(X_train: pd.DataFrame, y_train: np.ndarray, 
                model_params: dict = None) -> RandomForestClassifier:
    """
    Train the random forest model
    """
    if model_params is None:
        model_params = {
            'n_jobs': -1,
            'random_state': 42,
            'criterion': "gini",
            'n_estimators': 100,
            'verbose': False
        }
    
    clf = RandomForestClassifier(**model_params)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X: pd.DataFrame, y: np.ndarray, 
                   target_names: list = None) -> dict:
    """
    Evaluate model performance
    """
    predictions = model.predict(X)
    
    if target_names is None:
        target_names = ['Not Survived', 'Survived']
    
    report = metrics.classification_report(y, predictions, target_names=target_names)
    accuracy = metrics.accuracy_score(y, predictions)
    
    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    
    return {
        'predictions': predictions,
        'report': report,
        'accuracy': accuracy
    }

def map_sex_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map gender to a numerical value
    """
    df_copy = df.copy()
    df_copy['Sex'] = df_copy['Sex'].map({'female': 1, 'male': 0}).astype(int)
    return df_copy