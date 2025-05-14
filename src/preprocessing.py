import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def bin_severity(score):
    """Categorize severity score into 3 classes."""
    if score <= 3:
        return 0  # Low
    elif score <= 6:
        return 1  # Medium
    else:
        return 2  # High

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset:
    - Select features
    - Bin the target
    - Split train/test
    - Define ColumnTransformer for encoding
    """
    target = 'Target_Severity_Score'
    features = ['Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking',
                'Obesity_Level', 'Treatment_Cost_USD']
    categorical_features = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']

    # Bin continuous severity score into discrete class
    df['Severity_Class'] = df[target].apply(bin_severity)

    # Split
    X = df[features + categorical_features]
    y = df['Severity_Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor
