import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from flask import Flask, redirect, request, session, url_for
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Spotify app credentials
CLIENT_ID = '10fdaa3e01374aae924189ca712eaa23'
CLIENT_SECRET = '10f8bfdd848a448e87c8acf3c1cf1c20'
REDIRECT_URI = 'http://localhost:8888/callback'

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Set up Spotify OAuth
sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                         client_secret=CLIENT_SECRET,
                         redirect_uri=REDIRECT_URI,
                         scope=["playlist-modify-public", "playlist-modify-private"])

# Initialize Spotify API with Client Credentials (for track feature retrieval)
client_credentials = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials)

# Initialize neural network and LabelEncoder for mood prediction
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)
le = LabelEncoder()

# Function to get track features from Spotify
def get_track_features(track_id):
    features = sp.audio_features([track_id])[0]
    if features:
        return [features['danceability'], features['energy'], features['valence']]
    return None

# Function to get tracks for a specific mood
def get_tracks_for_mood(mood):
    results = sp.search(q=mood, type='track', limit=50)
    track_ids = [track['id'] for track in results['tracks']['items']]
    return track_ids

# Dummy data for training the model
moods = ['happy', 'sad', 'relaxed']
X = []
y = []

# Fetch features for each mood (simulate this with existing tracks)
for mood in moods:
    track_ids = get_tracks_for_mood(mood)
    for track_id in track_ids:
        features = get_track_features(track_id)
        if features:
            X.append(features)
            y.append(mood)

X = np.array(X)
y = np.array(y)

# Convert labels to numerical values
y_encoded = le.fit_transform(y)

# Train the model (for demonstration purposes)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Function to recommend songs based on predicted mood
def recommend_songs(predicted_mood):
    print(f"Songs recommended for mood: {predicted_mood}")
    track_ids = get_tracks_for_mood(predicted_mood)
    song_info = []
    for track_id in track_ids[:3]:
        track = sp.track(track_id)
        song_info.append({
            'song_name': track['name'],
            'artist': track['artists'][0]['name'],
            'url': track['external_urls']['spotify']
        })
    return song_info

# Flask Routes
@app.route('/')
def home():
    if not session.get("token_info"):
        return redirect(url_for("login"))
    return "Welcome! Go to /mood to create a playlist."

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    token_info = sp_oauth.get_access_token(request.args['code'])
    session["token_info"] = token_info
    return redirect(url_for("mood"))

@app.route('/mood', methods=["GET", "POST"])
def mood():
    # Check if the user is authenticated
    token_info = session.get("token_info")
    if not token_info:
        return redirect(url_for("login"))

    # Get the user's access token
    sp = spotipy.Spotify(auth=token_info['access_token'])

    # Get the user-selected mood and features from the form
    mood = request.args.get('mood')  # This is the mood selected by the user
    try:
        # Get danceability, energy, and valence from the form, using default values if missing
        danceability = float(request.args.get('danceability', 0.7))  # Default value if not provided
        energy = float(request.args.get('energy', 0.7))  # Default value if not provided
        valence = float(request.args.get('valence', 0.7))  # Default value if not provided
    except ValueError:
        # If any of the inputs can't be converted to float, return an error message
        return "Error: Invalid input values for danceability, energy, or valence. Please ensure they are numbers between 0 and 1."

    # Input features for the neural network
    user_input_features = [danceability, energy, valence]

    # Predict the mood using the trained neural network
    predicted_encoded = mlp.predict([user_input_features])
    predicted_mood = le.inverse_transform(predicted_encoded)[0]
    print(f"Predicted mood: {predicted_mood}")

    # Recommend songs based on the predicted mood
    recommended_songs = recommend_songs(predicted_mood)

    # Create a new playlist
    user_id = sp.current_user()['id']
    playlist = sp.user_playlist_create(user_id, f"{predicted_mood.capitalize()} Playlist", public=True)

    # Add recommended tracks to the playlist
    track_uris = [track['url'] for track in recommended_songs]
    sp.playlist_add_items(playlist['id'], track_uris)

    return f"Playlist created! Check your Spotify account for your {predicted_mood} playlist."

if __name__ == '__main__':
    app.run(debug=True, port=8888)
