import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = r"D:\iris-classifier"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# create outputs folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load Iris Dataset
# -----------------------------
iris = load_iris()
print(iris)
X = iris.data
y = iris.target



# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# -----------------------------
# Train Model
# -----------------------------
model = SVC()

model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

conf_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

plt.savefig(conf_matrix_path)
plt.close()


# -----------------------------
# Save Model
# -----------------------------
model_path = os.path.join(OUTPUT_DIR, "iris_model.joblib")

joblib.dump(model, model_path)


print("\nSaved Files:")
print("Model saved at:", model_path)
print("Confusion matrix saved at:", conf_matrix_path)