import pandas as pd
from spotipy.oauth2 import SpotifyOAuth
import spotipy

# Load tagged tracks
tracks = pd.read_csv("tagged_tracks.csv")

# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="playlist-modify-private"))

# Generate playlists for each mood
for mood in tracks['Predicted Mood'].unique():
    mood_tracks = tracks[tracks['Predicted Mood'] == mood]
    track_uris = mood_tracks['Track URI'].tolist()

    # Create playlist
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(user_id, f"{mood.capitalize()} Playlist", public=False)
    sp.playlist_add_items(playlist['id'], track_uris)
    print(f"Playlist created for mood: {mood}")