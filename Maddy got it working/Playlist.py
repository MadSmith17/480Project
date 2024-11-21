import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, redirect, request, session, url_for
import os

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

# Routes
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

@app.route('/mood')
def mood():
    # Check if the user is authenticated
    token_info = session.get("token_info")
    if not token_info:
        return redirect(url_for("login"))

    # Get the user's access token
    sp = spotipy.Spotify(auth=token_info['access_token'])

    # Get mood from user input (for now hardcoded)
    mood = "happy"  # Replace with dynamic input (form, dropdown, etc.)
    print(f"User mood: {mood}")

    # Map mood to Spotify audio features
    mood_params = {
        "happy": {"min_valence": 0.7, "min_energy": 0.6},
        "sad": {"min_valence": 0.3, "min_energy": 0.3},
        "relaxed": {"min_valence": 0.6, "min_energy": 0.4},
    }

    # Get recommendations based on mood
    recommendations = sp.recommendations(
        seed_genres=["pop"],  # You can use other genres or seeds
        limit=10,
        min_valence=mood_params[mood]["min_valence"],
        min_energy=mood_params[mood]["min_energy"]
    )

    # Create a new playlist
    user_id = sp.current_user()['id']
    playlist = sp.user_playlist_create(user_id, f"{mood.capitalize()} Playlist", public=True)

    # Add recommended tracks to the playlist
    track_uris = [track['uri'] for track in recommendations['tracks']]
    sp.playlist_add_items(playlist['id'], track_uris)

    return f"Playlist created! Check your Spotify account for your {mood} playlist."

if __name__ == '__main__':
    app.run(debug=True, port=8888)
