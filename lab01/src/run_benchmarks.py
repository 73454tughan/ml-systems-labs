import joblib, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

acc = [{"model": "LogisticRegression", "test_accuracy": f"{lr.score(X_test, y_test):.4f}"},
       {"model": "RandomForestClassifier", "test_accuracy": f"{rf.score(X_test, y_test):.4f}"}]
pd.DataFrame(acc).to_csv("lab01/results/baseline_accuracy.csv", index=False)
joblib.dump(rf, "lab01/results/model.joblib")
print("[INFO] Lab 01 executed successfully.")
