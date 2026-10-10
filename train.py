"""Train a Logistic Regression model for Chronic Kidney Disease and save it as a pickle."""
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/kidney_disease_clean_data.csv"
MODEL_PATH = "model.pkl"

df = pd.read_csv(DATA_PATH)

# Drop Specific Gravity (sg) so the model does not depend on it
df = df.drop(columns=["sg"])

# y = last column (classification), X = all other columns
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
feature_names = list(X.columns)

# NumPy arrays are used so the deployed app does not need pandas
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
)

# Scaling matters because features such as wc (~8000) and rc (~5) have very different ranges
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Refit on all data for the final model, then save with the feature order
model.fit(X.values, y.values)
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "features": feature_names}, f)

print(f"Saved {MODEL_PATH} with features: {feature_names}")
