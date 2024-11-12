from flask import Flask, render_template, request, jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify API setup
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='10fdaa3e01374aae924189ca712eaa23',
                                               client_secret='10f8bfdd848a448e87c8acf3c1cf1c20',
                                               redirect_uri="http://127.0.0.1:5000/callback",
                                               scope=["playlist-modify-public", "playlist-modify-private", "user-library-read"]))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_playlist', methods=['POST'])
def get_playlist():
    try:
        # Get mood from the frontend
        mood = request.json.get('mood')
        if not mood:
            return jsonify({'error': 'Mood is required'}), 400
        
        # Search for tracks based on the mood
        results = sp.search(q=mood, type='track', limit=10)
        track_ids = [track['id'] for track in results['tracks']['items']]
        
        if not track_ids:
            return jsonify({'error': 'No tracks found for this mood'}), 404
        
        # Create the playlist
        user_id = sp.current_user()['id']
        playlist = sp.user_playlist_create(user=user_id, name=f"{mood} Playlist", public=False)

        # Add tracks to the playlist
        sp.playlist_add_items(playlist['id'], track_ids)

        return jsonify({'playlist_url': f'https://open.spotify.com/playlist/{playlist["id"]}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
