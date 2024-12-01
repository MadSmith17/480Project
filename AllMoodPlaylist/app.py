from flask import Flask, request, jsonify
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import pandas as pd
from sklearn.externals import joblib

# Initialize Flask app
app = Flask(__name__)

# Spotify authentication
sp_oauth = SpotifyOAuth(
    client_id="YOUR_SPOTIFY_CLIENT_ID",
    client_secret="YOUR_SPOTIFY_CLIENT_SECRET",
    redirect_uri="http://localhost:5000/callback",
    scope="playlist-modify-private"
)
sp = None 

# Load models
mlp_model = joblib.load("mlp_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the AllMoodPlaylist Flask API!"})

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return jsonify({"auth_url": auth_url})

@app.route('/callback')
def callback():
    global sp
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    sp = spotipy.Spotify(auth=token_info['access_token'])
    return jsonify({"message": "Spotify authenticated successfully!"})

@app.route('/fetch-tracks', methods=['GET'])
def fetch_tracks():
    global sp
    if not sp:
        return jsonify({"error": "User is not authenticated with Spotify"}), 401

    results = sp.search(q="*", type="track", limit=50)  # Fetch 50 RANDOM tracks, this is the limit from spotify at the moment
    tracks = []

    for track in results['tracks']['items']:
        features = sp.audio_features([track['id']])[0]
        if features:
            track_data = {
                "Track Name": track['name'],
                "Artist": track['artists'][0]['name'],
                "Track URI": track['uri'],
                "danceability": features['danceability'],
                "energy": features['energy'],
                "valence": features['valence'],
                "acousticness": features['acousticness'],
                "tempo": features['tempo'],
                "speechiness": features['speechiness'],
                "liveness": features['liveness']
            }
            tracks.append(track_data)

    # Save raw tracks
    tracks_df = pd.DataFrame(tracks)
    tracks_df.to_csv("fetched_tracks.csv", index=False)

    # Preprocess features
    features = tracks_df[['danceability', 'energy', 'valence', 'acousticness', 'tempo', 'speechiness', 'liveness']]
    features_scaled = scaler.transform(features)

    # Predict moods
    predictions = mlp_model.predict(features_scaled)
    tracks_df['Predicted Mood'] = label_encoder.inverse_transform(predictions)

    # Save tagged tracks
    tracks_df.to_csv("tagged_tracks.csv", index=False)

    return jsonify({
        "message": "Tracks fetched and tagged successfully!",
        "preview": tracks_df.head(5).to_dict(orient='records')
    })

@app.route('/generate-playlist', methods=['POST'])
def generate_playlist():
    """Generate Spotify playlists based on moods"""
    global sp
    if not sp:
        return jsonify({"error": "User is not authenticated with Spotify"}), 401

    # Load tagged tracks
    try:
        tracks_df = pd.read_csv("tagged_tracks.csv")
    except FileNotFoundError:
        return jsonify({"error": "No tagged tracks found. Please fetch tracks first."}), 400

    moods = tracks_df['Predicted Mood'].unique()
    user_id = sp.current_user()["id"]

    for mood in moods:
        mood_tracks = tracks_df[tracks_df['Predicted Mood'] == mood]
        track_uris = mood_tracks['Track URI'].tolist()

        # Create playlist
        playlist = sp.user_playlist_create(user_id, f"{mood.capitalize()} Playlist", public=False)
        sp.playlist_add_items(playlist['id'], track_uris)

    return jsonify({"message": "Playlists created successfully!"})

if __name__ == "__main__":
    app.run(debug=True)