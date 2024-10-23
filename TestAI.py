# Maddy's Attempt
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Authenticate Spotify
client_credentials = SpotifyClientCredentials(client_id='10fdaa3e01374aae924189ca712eaa23', 
                                              client_secret='10f8bfdd848a448e87c8acf3c1cf1c20')
sp = spotipy.Spotify(client_credentials_manager=client_credentials)

# Function to get track features from Spotify
def get_track_features(track_id):
    features = sp.audio_features([track_id])[0]
    if features is not None:
        return [features['danceability'], features['energy'], features['valence']]
    return None

# Get track IDs for a specific mood
def get_tracks_for_mood(mood):
    results = sp.search(q=mood, type='track', limit=50)
    track_ids = [track['id'] for track in results['tracks']['items']]
    return track_ids

# Initialize lists to hold data
X = []
y = []

# Example moods to search for
moods = ['happy', 'sad', 'relaxed']

# Fetch features for each mood
for mood in moods:
    track_ids = get_tracks_for_mood(mood)
    for track_id in track_ids:
        features = get_track_features(track_id)
        if features:
            X.append(features)
            y.append(mood)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert target labels to numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training, validation, and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2 of total data

# Define the neural network
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)

# Custom training loop to log losses
train_losses = []
val_losses = []
for epoch in range(2000):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_encoded))
    
    # Record training loss
    if epoch % 100 == 0:
        train_loss = mlp.loss_
        train_losses.append(round(train_loss, 4))  # Round the loss for readability
        
        # Validate on validation set
        val_loss = np.mean(1 - mlp.predict_proba(X_val)[np.arange(len(y_val)), y_val])
        val_losses.append(round(val_loss, 4))

        print(f"Epoch [{epoch}/2000], Train Loss: {train_loss}, Val Loss: {val_loss}")

# Evaluate the model on the test set
y_pred = mlp.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs (x100)')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.legend()
plt.grid()
plt.show()
