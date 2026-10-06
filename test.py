import warnings
import joblib as j
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.datasets import load_iris
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
print("Libraries imported successfully")

# 1. Load data and properly include the target column
i = load_iris()
df = pd.DataFrame(i.data, columns=i.feature_names)
df['target'] = i.target  # <-- FIXED: Added the actual species labels (0, 1, 2)

print("Initial data snippet:")
print(df.head())
print('')

# 2. Outlier Removal (Calculate z-score only on feature columns)
feature_cols = i.feature_names
z = np.abs(zscore(df[feature_cols]))
dfn = df[(z < 3).all(axis=1)]
print("Old shape:", df.shape, "\nNew shape:", dfn.shape)
print('')

# 3. Correctly Split Features (x) and Target (y)
x = dfn[feature_cols]   # <-- FIXED: Contains only the 4 flower features
y = dfn['target']       # <-- FIXED: Contains the flower species classes

# 4. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print('x_train shape:', x_train.shape, '| y_train shape:', y_train.shape)

# 5. Initialize, Train, and Evaluate Model
svc = SVC()
svc.fit(x_train, y_train)

# Evaluate performance
predictions = svc.predict(x_test)
print(f"Accuracy Score: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 6. Save the trained model
j.dump(svc, 'svc.pkl')
print("Model Saved successfully as 'svc.pkl'")
