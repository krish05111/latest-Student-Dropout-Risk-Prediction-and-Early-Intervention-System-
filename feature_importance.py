import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load(
    "models/dropout_model.pkl"
)

# Feature names
features = [
    "attendance",
    "marks",
    "study_hours",
    "failed_subjects",
    "assignments_completed"
]

# Get feature importance
importance = abs(model.coef_[0])

# Create DataFrame
importance_data = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort by importance
importance_data = importance_data.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_data)

# Create graph
plt.figure(figsize=(8, 5))

plt.bar(
    importance_data["Feature"],
    importance_data["Importance"]
)

plt.xlabel("Student Factors")
plt.ylabel("Importance")

plt.title("Factors Affecting Dropout Risk")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()