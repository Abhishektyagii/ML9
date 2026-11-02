
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Create Project Folder
os.makedirs("models", exist_ok=True)

# Step 2: Generate Sample Loan Dataset
np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "Gender": np.random.choice(["Male", "Female"], n),
    "Married": np.random.choice(["Yes", "No"], n),
    "Education": np.random.choice(["Graduate", "Not Graduate"], n),
    "Self_Employed": np.random.choice(["Yes", "No"], n),
    "ApplicantIncome": np.random.randint(1500, 15000, n),
    "CoapplicantIncome": np.random.randint(0, 8000, n),
    "LoanAmount": np.random.randint(50, 500, n),
    "Loan_Amount_Term": np.random.choice([120, 180, 240, 300, 360], n),
    "Credit_History": np.random.choice([0, 1], n, p=[0.25, 0.75]),
    "Property_Area": np.random.choice(
        ["Urban", "Semiurban", "Rural"], n
    )
})

# Step 3: Create Target Column
approval_score = (
    (df["Credit_History"] * 4)
    + (df["ApplicantIncome"] > 5000).astype(int) * 2
    + (df["CoapplicantIncome"] > 2000).astype(int)
    + (df["Education"] == "Graduate").astype(int)
    - (df["LoanAmount"] > 350).astype(int) * 2
    + np.random.normal(0, 1.5, n)
)

df["Loan_Status"] = np.where(
    approval_score >= 4, "Approved", "Rejected"
)

# Step 4: Save Dataset
df.to_csv("loan_dataset.csv", index=False)

print("Dataset Created Successfully!")
print(df.head())

# Step 5: Separate Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

categorical_features = [
    "Gender", "Married", "Education",
    "Self_Employed", "Property_Area"
]

numeric_features = [
    "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term",
    "Credit_History"
]

# Step 6: Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Step 7: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 8: Compare ML Algorithms
algorithms = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, random_state=42
    )
}

best_model = None
best_accuracy = 0
best_name = ""

for name, algorithm in algorithms.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", algorithm)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n------------------------------")
    print("Algorithm:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# Step 9: Save Best Model
joblib.dump(best_model, "models/loan_model.joblib")

print("\nBest Algorithm:", best_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")
print("Trained Model Saved Successfully!")
