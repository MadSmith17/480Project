import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load processed DEAM data
data = pd.read_csv("processed_deam_annotations.csv")

# Define features (X) and target (y)
X = data[['arousal', 'valence']]
y = data['Mood']

# Encode moods
le = LabelEncoder()
y = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=2000, alpha=0.01, random_state=42)

# Train model
mlp.fit(X_train, y_train)

# Evaluate model
y_pred = mlp.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
import joblib
joblib.dump(mlp, "mlp_model.joblib")
print("Model saved!")