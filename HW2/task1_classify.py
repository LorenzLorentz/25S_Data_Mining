import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train_100000.csv')

# task 1.1
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

df[numerical_cols] = SimpleImputer(strategy='mean').fit_transform(df[numerical_cols])
df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['is_default'])
y = df['is_default']

mi_scores = mutual_info_classif(X, y, discrete_features='auto')
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
top20_features = mi_series.head(20).index.tolist()

plt.figure(figsize=(12, 12))
mi_series[top20_features].plot(kind='bar')
plt.title("Mutual Information Scores of Top 20 Features")
plt.ylabel("Mutual Information")
plt.xticks(rotation=330, ha='left', rotation_mode='anchor')
plt.tight_layout()
plt.savefig("task1.1_mutual_info.png")

# task 1.2
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_selected = df[top20_features].values

for model_name, model in models.items():
    print(f"Model: {model_name}")
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        auc_scores.append(auc)
        print(f"Fold {fold}: AUC = {auc:.4f}")

    print(f"Average AUC: {np.mean(auc_scores):.4f}\n")