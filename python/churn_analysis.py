import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Convert target column
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Convert text columns to numbers
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

feature_importance.head(10).to_csv(
    "../screenshots/top_features.csv",
    index=False
)

print("\nTop features saved successfully!")

import matplotlib.pyplot as plt

# Reload original dataset
df_original = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Create churn chart
df_original["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.tight_layout()

plt.savefig("../screenshots/churn_distribution.png")

print("Chart saved successfully!")