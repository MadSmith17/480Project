# 480 Semester Project: Mood.fm

<img src="Progress/moody.jpg" alt="Logo" width="400"/>

*AI-generated image created using Bing Image Creator on 10-28-2024.*

# Members:
* Carolina Mancina
* Madelyn Smith
* Taylor Peterson

# Project Overview:
We want to create a website that creates a playlist of recommended songs based on the users current mood, genres, and/or current favorite artists. Our goal is to curate better recommended playlists than other industry leading music platforms, such as Spotify or YouTube Music. Spotify only recommends based on listened or liked songs, and very rarely recommends new songs- we hope to improve upon that.

We would also like to add a rating system. Most music platforms lack the ability to rate songs based
on user input. Typically, there are only ways to ”like” a song- that is about it. Our idea would have
the user rate the song on a 5-star scale. We may also ask the user how they are feeling after listening to
a song, genre, album, or artist; providing a list of choices to choose from that we will create. This will
hopefully improve our algorithm and it’s accuracy to recommend songs.

# Dataset Overview
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

# Files
In the main branch, these files and directories exist:
* .vscode: Some needed materials for Visual Studio
* AllMoodPlaylist: This folder should be in the "Progress" folderm but is too large to be officially placed there.
* NewData/notebooks: Our **final** working program. Run **this** one!
* Progress: Contains all of the files we had worked on in the semester
* README.md: You are here!

# Getting Started
## Prerequisites
1. **Python Environment**: Ensure you have Python 3.8 or higher installed
2. **Dependencies**:
3. **Installation**: Clone this repository to your local machine.

## Project Structure

The project directoroy includes the following:
* notebooks/: Main project folder containing all resources and scripts
   * `data_preprocessing.ipynb`: Notebook for cleaning and preprocessing the dataset
   * `features_dataset.csv`: Dataset with extracted features for training the model.
   * `metadata_dataset.csv`; Metadata associated with the dataset
   * `mood_prediction_model.pth`: Pre-trained neural network model for mood classification.
   * `NN.ipynb`: Notebook for training the neural network.
   * `testing_model.ipynb`: Notebook for testing and evaluating the model.
   * `resources/dataset.csv`: Original raw dataset from Kaggle.

# References/Acknowledgments:
[1] [OpenAI. Chatgpt](https://chatgpt.com/?model=gpt-4o-mini): Gpt-4 model. Accessed: 2024

[2] [Spotify Web API](https://developer.spotify.com/documentation/web-api) to access and manage Spotify's music data (before it accesibility became limited)

[3] [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) Kaggle. Retrieved [11/28/2024]

[4] [Deam](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music): MediaEval Database for Emotional Analysis 2017. Dataset retrieved from Kaggle, as the official website was inaccessible. Anna Aljanaki, Mohammad Soleymani, and Frans Wiering. 

