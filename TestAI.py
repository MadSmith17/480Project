# Maddy's Attempt

import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Authenticate Spotify
client_credentials = SpotifyClientCredentials(client_id='YOUR_CLIENT_ID', client_secret='YOUR_CLIENT_SECRET')
sp = spotipy.Spotify(client_credentials_manager=client_credentials)

mood_keywords = ['happy', 'mad', 'sad', 'love', 'calm']
mood_labels = []
danceability = []
energy = []
valence = []
tempo = []
liveness = []
acousticness = []
speechiness = []

# Collect audio features from Spotify
for mood in mood_keywords:
    results = sp.search(q=mood, type='track', limit=50, market='US')
    tracks = results['tracks']['items']

    for track in tracks:
        track_id = track['id']
        
        # Retry logic for getting audio features
        for attempt in range(5):  # Try up to 5 times
            try:
                audio_features = sp.audio_features(track_id)[0]
                if audio_features:
                    mood_labels.append(mood_keywords.index(mood) + 1)
                    danceability.append(audio_features['danceability'])
                    energy.append(audio_features['energy'])
                    valence.append(audio_features['valence'])
                    tempo.append(audio_features['tempo'])
                    liveness.append(audio_features['liveness'])
                    acousticness.append(audio_features['acousticness'])
                    speechiness.append(audio_features['speechiness'])
                break  # Break if successful
            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:
                    print("Rate limit hit, waiting before retrying...")
                    time.sleep(5)  # Wait for 5 seconds before retrying
                else:
                    print(f"Error occurred: {e}")
                    break  # Break the loop for other errors

# Create DataFrame
data = {
    'mood': mood_labels,
    'danceability': danceability,
    'energy': energy,
    'valence': valence,
    'tempo': tempo,
    'liveness': liveness,
    'acousticness': acousticness,
    'speechiness': speechiness,
}
df = pd.DataFrame(data)

# Balance the dataset
min_samples = df['mood'].value_counts().min()
df_balanced = df.groupby('mood', group_keys=False).apply(lambda x: x.sample(n=min_samples, random_state=42)).reset_index(drop=True)

# Preprocess and target
X = df_balanced[['danceability', 'energy', 'valence', 'tempo', 'liveness', 'acousticness', 'speechiness']].values  
y = df_balanced['mood'].values  

# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train - 1)  # Adjust for zero-indexing
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test - 1)

# Define the neural network
class MoodNN(nn.Module):
    def __init__(self):
        super(MoodNN, self).__init__()
        self.fc1 = nn.Linear(7, 64)  # Adjust input size to 7 (number of features)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model, criterion, and optimizer
model = MoodNN()
criterion = nn.CrossEntropyLoss()  # Add class weights if needed
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
early_stopping_patience = 10
best_loss = float('inf')
patience_counter = 0

# Training loop
num_epochs = 1000  # Set an appropriate number of epochs
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Early stopping check
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Stop training if patience is exceeded
    if patience_counter >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Test Accuracy: {accuracy:.4f}')

# Save the trained model's parameters to a file
torch.save(model.state_dict(), 'mood_recommendation_model.pth')
