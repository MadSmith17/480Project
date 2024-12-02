
# Carolina For: Weekly #9

import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training, validation, and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

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

# Function to recommend three songs based on a predicted mood
def recommend_songs(predicted_mood):
    print(f"Songs recommended for mood: {predicted_mood}")
    track_ids = get_tracks_for_mood(predicted_mood)
    song_info = []
    
    for track_id in track_ids[:3]:  # Get the first 3 songs
        track = sp.track(track_id)
        song_info.append({
            'song_name': track['name'],
            'artist': track['artists'][0]['name'],
            'url': track['external_urls']['spotify']
        })
    
    return song_info

# Function to predict mood and recommend songs based on the user input mood
def predict_and_recommend(user_input_features):
    # Predict the mood based on user input features using the trained neural network
    predicted_encoded = mlp.predict([user_input_features])  # This will predict the encoded label
    predicted_mood = le.inverse_transform(predicted_encoded)[0]  # Convert back to the original label
    
    # Get song recommendations based on predicted mood
    recommended_songs = recommend_songs(predicted_mood)
    
    # Print the recommended songs
    for idx, song in enumerate(recommended_songs, 1):
        print(f"Song {idx}: {song['song_name']} by {song['artist']}")
        print(f"Listen here: {song['url']}")
        print()

# Ask the user to input their mood (this is a simplified version, replace with your preferred input method)
def get_user_input_features():
    print("\nPlease provide your mood features:")
    print("Danceability: (0 to 1)")
    danceability = float(input("Enter Danceability: "))
    print("Energy: (0 to 1)")
    energy = float(input("Enter Energy: "))
    print("Valence: (0 to 1)")
    valence = float(input("Enter Valence: "))
    
    return [danceability, energy, valence]

# Main function to interact with the user and get recommendations
def main():
    print("Welcome to the mood-based song recommendation system!")
    
    # Ask the user for features of the song (the features the neural network was trained on)
    user_input_features = get_user_input_features()
    
    # Predict and recommend songs based on the features
    predict_and_recommend(user_input_features)

# Run the main function
if __name__ == "__main__":
    main()