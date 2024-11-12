import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        return [
            features['danceability'], features['energy'], features['valence'], 
            features['tempo'], features['loudness'], features.get('acousticness', 0), 
            features.get('instrumentalness', 0), features.get('speechiness', 0)
        ]
    return None

# Get track IDs for a specific mood, fetching in batches to handle API limit
def get_tracks_for_mood(mood, total_tracks=200):
    track_ids = []
    limit = 50
    offset = 0

    while len(track_ids) < total_tracks:
        results = sp.search(q=mood, type='track', limit=limit, offset=offset)
        new_track_ids = [track['id'] for track in results['tracks']['items']]
        track_ids.extend(new_track_ids)

        offset += limit
        if len(new_track_ids) < limit:
            break

    return track_ids[:total_tracks]

# Initialize lists to hold data
X = []
y = []

# Expanded list of moods
moods = ['happy', 'sad', 'relaxed', 'angry', 'energetic', 'calm', 'romantic', 'melancholic']

# Fetch features for each mood
for mood in moods:
    track_ids = get_tracks_for_mood(mood, total_tracks=200)
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

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training, validation, and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Define a more complex neural network
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64), max_iter=2000, random_state=42, alpha=0.0001, 
    learning_rate_init=0.001, activation='relu'
)

# Custom training loop with early stopping
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 200
wait = 0

for epoch in range(2000):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_encoded))
    
    if epoch % 100 == 0:
        train_loss = mlp.loss_
        train_losses.append(round(train_loss, 4))
        
        val_loss = np.mean(1 - mlp.predict_proba(X_val)[np.arange(len(y_val)), y_val])
        val_losses.append(round(val_loss, 4))
        
        print(f"Epoch [{epoch+1}/2000], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping due to no improvement in validation loss.")
            break

# Evaluate the model on the test set
y_pred = mlp.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(train_losses)*100, 100), train_losses, label='Train Loss', color='blue')
plt.plot(range(0, len(val_losses)*100, 100), val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.legend()
plt.grid()
plt.show()
