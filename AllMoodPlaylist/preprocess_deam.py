import pandas as pd

# Load DEAM data
df1 = pd.read_csv("Resources/DEAM_Annotations/annotations/song_level/static_annotations_averaged_songs_1_2000.csv")
df2 = pd.read_csv("Resources/DEAM_Annotations/annotations/song_level/static_annotations_songs_2000_2058.csv")

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Normalize arousal and valence
df['arousal'] = (df['arousal'] - df['arousal'].min()) / (df['arousal'].max() - df['arousal'].min())
df['valence'] = (df['valence'] - df['valence'].min()) / (df['valence'].max() - df['valence'].min())

# Assign moods based on arousal and valence
def assign_mood(row):
    if row['arousal'] > 0.5 and row['valence'] > 0.5:
        return "Energetic"
    elif row['arousal'] > 0.5 and row['valence'] <= 0.5:
        return "Angry"
    elif row['arousal'] <= 0.5 and row['valence'] > 0.5:
        return "Happy"
    else:
        return "Sad"

df['Mood'] = df.apply(assign_mood, axis=1)

# Save processed data
df.to_csv("processed_deam_annotations.csv", index=False)
print("Processed DEAM data saved!")