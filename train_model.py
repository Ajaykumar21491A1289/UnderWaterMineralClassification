import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import pickle


def load_data():
    main_df = pd.read_csv('sonar.csv', header=None)
    X = main_df.drop(columns=[60])
    y = main_df[60].apply(lambda x: 1 if x == 'M' else 0)
    return X, y


def save_model(model):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)


def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Create AdaBoostClassifier with default base estimator
    ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)

    ada_boost.fit(X_train_resampled, y_train_resampled)
    save_model(ada_boost)


# Recreate the pickle file
train_model()
