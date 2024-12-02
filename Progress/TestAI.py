from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
from flask_cors import CORS

# Load environment variables from a .env file
load_dotenv()

# Spotify API setup with environment variables
sp_oauth = SpotifyOAuth(
    client_id=os.getenv('10fdaa3e01374aae924189ca712eaa23'),
    client_secret=os.getenv('10f8bfdd848a448e87c8acf3c1cf1c20'),
    redirect_uri="http://127.0.0.1:5000/callback",
    scope=["playlist-modify-public", "playlist-modify-private", "user-library-read"]
)

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key')  # Replace with your secret key
app.config['SESSION_COOKIE_NAME'] = 'Spotify-Session'

@app.route('/')
def index():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    
    # Save token info in session
    session['token_info'] = token_info
    return redirect(url_for('home'))

@app.route('/home')
def home():
    token_info = session.get('token_info', None)
    if not token_info:
        return redirect('/')
    
    sp = spotipy.Spotify(auth=token_info['access_token'])
    return render_template('index.html')

@app.route('/get_playlist', methods=['POST'])
def get_playlist():
    try:
        token_info = session.get('token_info', None)
        if not token_info:
            return jsonify({'error': 'User not authenticated'}), 401

        sp = spotipy.Spotify(auth=token_info['access_token'])

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
