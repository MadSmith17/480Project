import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# Load fetched Spotify tracks
tracks = pd.read_csv("fetched_tracks.csv")

# Load the trained neural network
mlp = joblib.load("mlp_model.joblib")

# Load and fit scaler with DEAM data
deam_data = pd.read_csv("processed_deam_annotations.csv")
scaler = StandardScaler()
scaler.fit(deam_data[['arousal', 'valence']])

# Standardize Spotify track features
features = tracks[['danceability', 'energy', 'valence', 'acousticness', 'tempo', 'speechiness', 'liveness']]
features_scaled = scaler.transform(features)

# Predict moods
predictions = mlp.predict(features_scaled)
tracks['Predicted Mood'] = predictions

# Save tagged tracks
tracks.to_csv("tagged_tracks.csv", index=False)
print("Tracks tagged with moods!")