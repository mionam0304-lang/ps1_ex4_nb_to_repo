# %% [markdown]
# # Titanic Data Analysis Journey
# 
# This notebook presents a comprehensive analysis of the Titanic dataset, 
# from data loading to model building and evaluation.

# %% [markdown]
# # Prepare for analysis

# %% [markdown]
# ## Load packages and custom modules

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
import sys
sys.path.append('../src')

from data_loader import load_data, combine_datasets, get_data_info
from data_analyzer import analyze_missing_data, analyze_frequent_data, analyze_unique_values
from feature_engineer import (create_family_size, create_age_intervals, create_fare_intervals,
                             create_sex_pclass_feature, parse_names, create_family_type, unify_titles)
from visualization import plot_count_pairs, plot_distribution_pairs
from model import prepare_features, train_model, evaluate_model, map_sex_to_numeric

# %%
# Configuration parameters
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
VALID_SIZE = 0.2
RANDOM_STATE = 42

# %% [markdown]
# ## Read the data

# %%
# Load data using our custom function
train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

# %% [markdown]
# # Preliminary data inspection

# %% [markdown]
# ## Quick glimpse of the data

# %%
get_data_info(train_df)

# %%
get_data_info(test_df)

# %% [markdown]
# ## Statistical analysis of the data

# %% [markdown]
# ### Missing data analysis

# %%
missing_train = analyze_missing_data(train_df)
print("Training data missing values:")
print(missing_train)

# %%
missing_test = analyze_missing_data(test_df)
print("Test data missing values:")
print(missing_test)

# %% [markdown]
# ### Most frequent data

# %%
frequent_train = analyze_frequent_data(train_df)
print("Training data frequent values:")
print(frequent_train)

# %%
frequent_test = analyze_frequent_data(test_df)
print("Test data frequent values:")
print(frequent_test)

# %% [markdown]
# ### Unique values analysis

# %%
unique_train = analyze_unique_values(train_df)
print("Training data unique values:")
print(unique_train)

# %%
unique_test = analyze_unique_values(test_df)
print("Test data unique values:")
print(unique_test)

# %% [markdown]
# # Exploratory data analysis

# %% [markdown]
# ## Univariate analysis for all features

# %%
# Combine datasets for visualization
all_df = combine_datasets(train_df, test_df)

# %%
# Plot count pairs for categorical features
plot_count_pairs(all_df, "Sex", hue_column="set", 
                 title="Number of passengers / Sex (Train vs Test)")

# %%
# Plot distribution with survival hue
plot_distribution_pairs(train_df, "Sex", hue_column="Survived",
                       title="Number of passengers / Sex (by Survival)")

# %%
# Plot multiple categorical features
categorical_features = ["Sex", "Pclass", "SibSp", "Parch", "Embarked"]
for feature in categorical_features:
    plot_count_pairs(all_df, feature, hue_column="set",
                    title=f"Number of passengers / {feature}")

# %%
# Plot distribution for numerical features
numerical_features = ["Age", "Fare"]
for feature in numerical_features:
    plot_distribution_pairs(train_df, feature, hue_column="Survived",
                           title=f"Distribution of {feature} (by Survival)")

# %% [markdown]
# ## Feature Engineering

# %%
# Apply feature engineering to all datasets
all_df = create_family_size(all_df)
train_df = create_family_size(train_df)

all_df = create_age_intervals(all_df)
train_df = create_age_intervals(train_df)

all_df = create_fare_intervals(all_df)
train_df = create_fare_intervals(train_df)

all_df = create_sex_pclass_feature(all_df)
train_df = create_sex_pclass_feature(train_df)

# %%
# Parse names
all_df = parse_names(all_df)
train_df = parse_names(train_df)

# %%
# Create family type and unify titles
all_df = create_family_type(all_df)
train_df = create_family_type(train_df)

all_df = unify_titles(all_df)
train_df = unify_titles(train_df)

# %%
# Plot engineered features
plot_count_pairs(all_df, "Family Size", hue_column="Survived",
                title="Family Size Distribution (by Survival)")

plot_count_pairs(all_df, "Age Interval", hue_column="Survived",
                title="Age Interval Distribution (by Survival)")

# %% [markdown]
# ## Multivariate analysis

# %%
# Multivariate visualizations
plot_count_pairs(all_df, "Age Interval", hue_column="Pclass",
                title="Age Interval by Passenger Class")

plot_count_pairs(all_df, "Pclass", hue_column="Fare Interval",
                title="Passenger Class by Fare Interval")

# %% [markdown]
# # Baseline Model

# %% [markdown]
# ## Feature engineering for modeling

# %%
# Map categorical features to numerical
train_df = map_sex_to_numeric(train_df)
test_df = map_sex_to_numeric(test_df)

# %% [markdown]
# ## Create train-validation split

# %%
VALID_SIZE = 0.2
train, valid = train_test_split(train_df, test_size=VALID_SIZE, 
                               random_state=RANDOM_STATE, shuffle=True)

# %% [markdown]
# ## Define features and train model

# %%
predictors = ["Sex", "Pclass"]
target = 'Survived'

# Prepare features
train_X, train_Y = prepare_features(train, predictors, target)
valid_X, valid_Y = prepare_features(valid, predictors, target)

# %%
# Train model
clf = train_model(train_X, train_Y)

# %%
# Evaluate on training data
print("Training Data Evaluation:")
train_results = evaluate_model(clf, train_X, train_Y)

# %%
# Evaluate on validation data
print("Validation Data Evaluation:")
valid_results = evaluate_model(clf, valid_X, valid_Y)

# %% [markdown]
# # Conclusion
# 
# This analysis demonstrates a complete workflow from data loading to model building using modular, reusable functions.