import numpy as np
import pandas as pd
from pandas.core.series import Series
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

def preprocess(df:DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

def calc_mi(X:DataFrame, y:Series) -> Series:
    mi_scores = mutual_info_classif(X, y, discrete_features='auto')
    mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return mi_scores

def plot(scores:Series, features:list[str]) -> None:
    plt.figure(figsize=(12, 12))
    scores[features].plot(kind='bar')
    plt.title("Mutual Information Scores of Top 20 Features")
    plt.ylabel("Mutual Information")
    plt.xticks(rotation=330, ha='left', rotation_mode='anchor')
    plt.tight_layout()
    plt.savefig("task1.1_mutual_info.png")

def train(model, model_name:str, X:np.ndarray, y:Series, skf) -> None:
    print(f"Model: {model_name}")
    
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        auc_scores.append(auc)
        print(f"Fold {fold}: AUC = {auc}")

    print(f"Average AUC: {np.mean(auc_scores)}\n")

def main():
    df = pd.read_csv('train_100000.csv')
    preprocess(df)
    
    X = df.drop(columns=['is_default'])
    y = df['is_default']
    mi_scores = calc_mi(X, y)
    top20_features = mi_scores.head(20).index.tolist()

    plot(mi_scores, top20_features)

    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_selected = df[top20_features].values

    for model_name, model in models.items():
        train(model, model_name, X_selected, y, skf)

if __name__ == "__main__":
    main()