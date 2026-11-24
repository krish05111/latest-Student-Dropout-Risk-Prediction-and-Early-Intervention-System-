import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/student_data.csv")

features = [
    "attendance",
    "marks",
    "study_hours",
    "failed_subjects",
    "assignments_completed"
]

X = data[features]
y = data["dropout"]

# Use the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model
model = joblib.load("models/dropout_model.pkl")

# Predict
predictions = model.predict(X_test)

# Classification report
print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Confusion matrix
matrix = confusion_matrix(
    y_test,
    predictions
)

print("\nCONFUSION MATRIX")
print(matrix)

# Display graph
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["No Dropout", "Dropout"]
)

display.plot()

plt.title("Dropout Prediction Confusion Matrix")

plt.show()