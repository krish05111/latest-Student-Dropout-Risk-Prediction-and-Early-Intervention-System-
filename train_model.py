import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Input features
features = [
    "attendance",
    "marks",
    "study_hours",
    "failed_subjects",
    "assignments_completed"
]

X = data[features]
y = data["dropout"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models to compare
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}

results = []

best_model = None
best_score = 0
best_model_name = ""

# Train and evaluate every model
for name, model in models.items():

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # Display results
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))

    # Save results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Select best model using F1 score
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = name

# Create comparison table
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df)

# Save comparison results
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

# Save best model
joblib.dump(
    best_model,
    "models/dropout_model.pkl"
)

print("\nBest Model:", best_model_name)
print("Best model saved successfully!")