print("🚀 Starting training...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load data
df = pd.read_csv('data/patients.csv')
print("✅ Data loaded")

# Features and target
X = df.drop(columns=['risk_label'])
y = df['risk_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model trained")

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Model Accuracy: {acc:.2f}")
print("Confusion Matrix:\n", cm)

# Save model to file
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved to model/model.pkl")

