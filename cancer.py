import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

standard_scaler = StandardScaler()
model = {
    "logistic regression": make_pipeline(standard_scaler, LogisticRegression()),
    "decision tree": make_pipeline(
        standard_scaler, DecisionTreeClassifier(random_state=42)
    ),
    "random forest": make_pipeline(
        standard_scaler, RandomForestClassifier(random_state=42)
    ),
}
for name, m in model.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    scores = cross_val_score(m, X, y, cv=5)
    report = classification_report(y_test, preds, target_names=data.target_names)
    print(
        f"{name} Accuracy Score: {accuracy_score(y_test, preds):.2f}"
        f"\n Confusion Matrix:{confusion_matrix(y_test, preds)}\n"
        f"Mean Score: {scores.mean():.2f} | std: {scores.std():.2f}\n"
        f"Classification Report: {report}"
    )
rf = model["random forest"].named_steps["randomforestclassifier"]
feature_names = data.feature_names
rf_importances = rf.feature_importances_
series = pd.Series(rf_importances, feature_names).sort_values(ascending=False)
feats = series.head(10).plot(kind="barh", title="Top 10 Features Importance")

plt.figure(figsize=(10, 6))
plt.barh(series.index, series.values)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
