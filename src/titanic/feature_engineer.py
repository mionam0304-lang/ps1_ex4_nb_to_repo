import pandas as pd
import numpy as np

def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the family size feature based on SibSp and Parch
    """
    df_copy = df.copy()
    df_copy["Family Size"] = df_copy["SibSp"] + df_copy["Parch"] + 1
    return df_copy

def create_age_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide the age into intervals
    """
    df_copy = df.copy()
    df_copy["Age Interval"] = 0.0
    df_copy.loc[df_copy['Age'] <= 16, 'Age Interval'] = 0
    df_copy.loc[(df_copy['Age'] > 16) & (df_copy['Age'] <= 32), 'Age Interval'] = 1
    df_copy.loc[(df_copy['Age'] > 32) & (df_copy['Age'] <= 48), 'Age Interval'] = 2
    df_copy.loc[(df_copy['Age'] > 48) & (df_copy['Age'] <= 64), 'Age Interval'] = 3
    df_copy.loc[df_copy['Age'] > 64, 'Age Interval'] = 4
    return df_copy

def create_fare_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide the ticket prices into intervals
    """
    df_copy = df.copy()
    df_copy['Fare Interval'] = 0.0
    df_copy.loc[df_copy['Fare'] <= 7.91, 'Fare Interval'] = 0
    df_copy.loc[(df_copy['Fare'] > 7.91) & (df_copy['Fare'] <= 14.454), 'Fare Interval'] = 1
    df_copy.loc[(df_copy['Fare'] > 14.454) & (df_copy['Fare'] <= 31), 'Fare Interval'] = 2
    df_copy.loc[df_copy['Fare'] > 31, 'Fare Interval'] = 3
    return df_copy

def create_sex_pclass_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite features of gender and cabin class
    """
    df_copy = df.copy()
    df_copy["Sex_Pclass"] = df_copy.apply(
        lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1
    )
    return df_copy

def parse_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the name, extract the family name, title, given name and maiden name
    """
    def _parse_single_name(row):
        try:
            text = row["Name"]
            split_text = text.split(",")
            family_name = split_text[0]
            next_text = split_text[1]
            split_text = next_text.split(".")
            title = (split_text[0] + ".").lstrip().rstrip()
            next_text = split_text[1]
            
            if "(" in next_text:
                split_text = next_text.split("(")
                given_name = split_text[0]
                maiden_name = split_text[1].rstrip(")")
                return pd.Series([family_name, title, given_name, maiden_name])
            else:
                given_name = next_text
                return pd.Series([family_name, title, given_name, None])
        except Exception as ex:
            print(f"An error occurred when parsing the name: {ex}")
            return pd.Series([None, None, None, None])
    
    df_copy = df.copy()
    df_copy[["Family Name", "Title", "Given Name", "Maiden Name"]] = df_copy.apply(
        _parse_single_name, axis=1
    )
    return df_copy

def create_family_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create family type features based on family size
    """
    df_copy = df.copy()
    df_copy["Family Type"] = df_copy["Family Size"]
    df_copy.loc[df_copy["Family Size"] == 1, "Family Type"] = "Single"
    df_copy.loc[(df_copy["Family Size"] > 1) & (df_copy["Family Size"] < 5), "Family Type"] = "Small"
    df_copy.loc[(df_copy["Family Size"] >= 5), "Family Type"] = "Large"
    return df_copy

def unify_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify and aggregate titles
    """
    df_copy = df.copy()
    df_copy["Titles"] = df_copy["Title"]
    
    # Unify general titles
    df_copy['Titles'] = df_copy['Titles'].replace('Mlle.', 'Miss.')
    df_copy['Titles'] = df_copy['Titles'].replace('Ms.', 'Miss.')
    df_copy['Titles'] = df_copy['Titles'].replace('Mme.', 'Mrs.')
    
    # Unify special titles
    rare_titles = ['Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 
                   'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.']
    df_copy['Titles'] = df_copy['Titles'].replace(rare_titles, 'Rare')
    
    return df_copy