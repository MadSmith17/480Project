import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Spotify API authentication
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv('SPOTIPY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIPY_CLIENT_SECRET')
))

def fetch_random_tracks(limit=50):

    # Fetch a random sample of tracks from Spotify and extract their audio features.

    print("Fetching random tracks...")
    all_tracks = []
    try:
        # Use a **wildcard search query**
        results = sp.search(q="*", type="track", limit=limit)
        for track in results['tracks']['items']:
            track_id = track['id']
            features = sp.audio_features([track_id])[0]
            if features:
                all_tracks.append({
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
                })
        # Save to CSV
        pd.DataFrame(all_tracks).to_csv("fetched_tracks.csv", index=False)
        print("Random tracks saved to fetched_tracks.csv.")
    except Exception as e:
        print(f"Error fetching tracks: {e}")

if __name__ == "__main__":
    fetch_random_tracks(limit=50) #spotify limits