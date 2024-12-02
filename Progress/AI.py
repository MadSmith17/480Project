import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#authenticate Spotify
client_credentials = SpotifyClientCredentials(client_id='10fdaa3e01374aae924189ca712eaa23', client_secret='10f8bfdd848a448e87c8acf3c1cf1c20')
sp = spotipy.Spotify(client_credentials_manager=client_credentials)


mood_keywords = ['happy', 'mad', 'sad', 'love', 'calm']
mood_labels = []
danceability = []
energy = []
valence = []

#should we inlcude genres???
genres = []


#check mood and loop through songs
#should we change limit??
#how to get user input???
for mood in mood_keywords:
    results = sp.search(q=mood, type='track', limit=50)
    tracks = results['tracks']['items']

    for track in tracks:
        track_id = track['id']
        audio_features = sp.audio_features(track_id)[0]
        track_genres = track['album']['genres'] if 'genres' in track['album'] else []

        if audio_features:
            mood_labels.append(mood_keywords.index(mood) + 1)  
            danceability.append(audio_features['danceability'])
            energy.append(audio_features['energy'])
            valence.append(audio_features['valence'])
            genres.append(track_genres)


#DateFRAme
data = {
    'mood': mood_labels,
    'danceability': danceability,
    'energy': energy,
    'valence': valence,
    'genres': genres,
}

df = pd.DataFrame(data)

#for testing we can comment the code and print data to see whats up

#preprocess and target
X = df[['danceability', 'energy', 'valence']].values  
y = df['mood'].values  

#training and testing sets
#we might need to change this
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale features
#we might also need to change this idk
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#tensors
X_train_tensor = torch.FloatTensor(X_train)
#we might need to adjust this v
y_train_tensor = torch.LongTensor(y_train - 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test - 1)

#our neural  network
# Adjust architecture
class MoodNN(nn.Module):
    def __init__(self):
        super(MoodNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
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




model = MoodNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#training loop

#we might have to change the number of epochs
num_epochs = 100000

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    #prints once every ten (we will need to readjust for number of epochs)
    if (epoch + 1) % 10000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

#evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Test Accuracy: {accuracy:.4f}')


#idk if we need to save it
# save the trained neural network model's parameters to a file
torch.save(model.state_dict(), 'mood_recommendation_model.pth')
