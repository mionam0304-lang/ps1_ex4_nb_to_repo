"""
Titanic Data Analysis Package

A modular package for analyzing Titanic passenger data and predicting survival.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import load_data, combine_datasets, get_data_info
from .data_analyzer import analyze_missing_data, analyze_frequent_data, analyze_unique_values
from .feature_engineer import (create_family_size, create_age_intervals, create_fare_intervals,
                              create_sex_pclass_feature, parse_names, create_family_type, unify_titles)
from .visualization import plot_count_pairs, plot_distribution_pairs
from .model import prepare_features, train_model, evaluate_model, map_sex_to_numeric