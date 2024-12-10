# Mood.fm - 480 Project

<img src="Progress/moody.jpg" alt="Logo" width="400"/>

*AI-generated image created using Bing Image Creator on 10-28-2024.*

---

## Table of Contents
1. [Members](#members)
2. [Project Overview](#project-overview)
   * [Why Mood.fm?](#why-moodfm)
   * [Goal](#goal)
   * [Challenges and Adaptations](#challenges-and-adaptations)
3. [Project Directory](#project-directory)
4. [Getting Started](#getting-started)
   * [Prerequisites](#prerequisites)
5. [Project Structure](#project-structure)
6. [Steps to Run the Project](#steps-to-run-the-project)
   * [Step 1: Clone the Repository](#step-1-clone-the-repository)
   * [Step 2: Preprocess the Data (Optional)](#step-2-preprocess-the-data-optional)
   * [Step 3: Train the Neural Network (Optional)](#step-3-train-the-neural-network-optional)
   * [Step 4: Test the Model and Generate Playlists](#step-4-test-the-model-and-generate-playlists)
7. [Analysis and Model Details](#analysis-and-model-details)
   * [Dataset and Preprocessing](#dataset-and-preprocessing)
   * [Neural Network Architecture](#neural-network-architecture)
   * [Training Process](#training-process)
   * [Model Testing and Playlist Generation](#model-testing-and-playlist-generation)
8. [Future Considerations](#future-considerations)
9. [References/Acknowledgments](#referencesacknowledgments)

---

# Members:
* Carolina Mancina
* Madelyn Smith
* Taylor Peterson

---

## Project Overview
We aim to create a website that generates playlists of recommended songs based on the user's current mood, genres, and/or favorite artists. Our goal is to outperform platforms like Spotify or YouTube Music by curating better recommendations, especially for discovering new songs.

### Why Mood.fm?
Spotify typically recommends songs based only on what users have listened to or liked, rarely introducing new music. Mood.fm seeks to improve upon this by leveraging moods and user input to create more personalized recommendations.

### Goal:
1. **Playlist Recommendations**: Curated by mood.
2. **Rating System**: Allows users to rate songs on a 5-star scale.
3. **Mood Tracking**: Users can provide feedback on their mood after listening to songs, genres, or albums to further refine recommendations.

### Challenges and Adaptations:
- Initially, we incorporated Spotify's Web API to fetch real-time music data. Unfortunately, due to recent changes in Spotify's API accessibility ([announcement](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api), [The Verge](https://www.theverge.com/2024/12/5/24311523/spotify-locked-down-apis-developers)), we shifted to using the Kaggle dataset instead.

---

## Project Directory:
* **`.vscode`**: Some needed materials for Visual Studio
* **`AllMoodPlaylist`**: This folder should be in the "Progress" folderm but is too large to be officially placed there.
* **`NewData/notebooks`**: Our **final** working program. Clone and run **this** one!
* **`Progress`**: Contains all of the files we have worked on in the semester.
* **`README.md`**: You are here!

---

## Getting Started

### Prerequisites:
1. **Python Environment**: Ensure Python 3.8 or higher is installed
2. **Required Libraries**:
   * `numpy`
   * `pandas`
   * `scikit-learn`
   * `seaborn`
   * `torch`
   * `torchvision`
   * `tqdm`

---

## Project Structure

The project directory includes the following:

`NewData/notebooks/`: Main project folder containing all resources and scripts:

   * `data_preprocessing.ipynb`: Notebook for cleaning and preprocessing the dataset
   * `features_dataset.csv`: Dataset with extracted features for training the model.
   * `metadata_dataset.csv`; Metadata associated with the dataset
   * `mood_prediction_model.pth`: Pre-trained neural network model for mood classification.
   * `NN.ipynb`: Notebook for training the neural network.
   * `testing_model.ipynb`: Notebook for testing and evaluating the model.
   * `resources/dataset.csv`: Original raw dataset from Kaggle.
 
## Steps to Run the Project:

### Step 1: Clone the repository

1. Clone the repository to your local machine and open the project folder in VSCode.

### Step 2: Preprocess the Data (Optional)

1. Open `data_preprocessing.ipynb`.
  
2. Run all cells to:
     * Clean the raw dataset (`resources/dataset.csv`) and generate:
         * `cleaned_dataset.csv`: Preprocessed dataset.
         * `features_dataset.csv`: Extracted features for training. 
         * `metadata_dataset.csv`: Metadata for analysis
           
4. Confirm the generated files are in the `notebooks` directory. 

***Note**: This step is optional as these files already exist.*

---

### Step 3: Train the Neural Network (Optional)

   1. Open `NN.ipynb`.
   
   2. Run all cells to:
      * Load `features_dataset.csv`.
      * Train the neural network for mood clasification.
      * Save outputs:
        * `mood_predicting_model.pth`: Model weights
        * `label_encoder.pkl`: Encoded mood labels
        * `scaler.pkl`: Scaler for feature normalization
   
*This will overwrite `mood_prediction_model.pth` with a newly trained model.*

---

### Step 4: Test the Model and Generate Playlists

#### Testing

1. Open `testing_model.ipynb`.
   
2. Run all cells to:
   * Normalize features using scaler.pkl.
   * Predict moods uing the saved model.
   * Generate and print playlists based on predicted moods.

3. View:
   * Performance Metrics: Classification report and confusion matrix
   * Feature pairplot and distribution charts.

#### Generate Playlists:
1. Locate the function `generate_user_playlist()` underneath the section labeled "Playlist Generation" (at the bottom)
2. Once running, user is prompted to input a number corresponding to a specific mood
  * **1**: Happy
  * **2**: Sad
  * **3**: Calm
  * **4**: Energetic
4. Choose one of the above, and a playlist of (10) songs will be generated.

### Example User Interaction:

<img src="Progress/exampleoutput.png" alt="output"/>

---

## Analysis and Model Details

### Introduction to Features
* **Valence**: Measures positivity in a track (e.g., happy vs. sad).
* **Energy**: Reflects intensity and activity (e.g., calm vs. energetic).
* **Tempo**: Measures speed, contributing to mood categorization.

*These features were chosen for their relevance to mood classification.*

### Dataset and Preprocessing
#### **Dataset**:
  * **Source**: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
  * **Size**: Contains 114,000 Spotify tracks and their attributes.
    
####  **Preprocessing Steps**:
1. **Feature Selection**:
   * Extracted key attributes required for mood classification (e.g., valence, energy, tempo, danceability).
2. **Mood Encoding**:
   * Assigned moods based on thresholds for valence and energy (e.g., valence > 0.5 = happy).
3. **Normalization**:
   * Applied scaler.pkl to ensure features were on consistent scales for neural network input.
4. **Label Mapping**:
   * Encoded mood categories numerically and saved the mapping in label_encoder.pkl.
5. **Dataset Creation**:
   * Generated separate datasets for training and evaluation.
6. **Data Storage**:
   * Saved processed data into:
      - `features_dataset.csv`: For training the neural network.
      - `metadata_dataset.csv`: For additional analysis and metadata.

---

### Neural Network Architecture:
* **Input Layer**: 12 features
* **Hidden Layers**:
    * First Layer: 64 neurons, ReLU activation.
    * Second Layer: 32 neurons, ReLU activation
* **Output Layer**: 4 neurons, representing the four mood categories: Happy, Sad, Calm, and Energetic.

---

### Training Process
1. **Training Dataset**: 80% training, 20% testing split.
2. **Forward Pass**:
   * Input: Feature vector from the dataset (e.g., valence, tempo).
   * Output: Predicted probabilities for each mood category.
3. **Loss Calculation**:
   * Used CrossEntropyLoss to calculate the error between predicted and actual moods.
4. **Backpropagation**:
   * Calculated gradients to adjust the weights of the network.
   * Optimized using Adam optimizer.
5. **Evaluation Metrics**:
   * Achieved a test accuracy of **99.37%**.
   * Generated accuracy and loss graphs to track performance.

### Model Testing and Playlist Generation
1. **Normalization**: Scaled features in `metadata_dataset.csv` using `scaler.pkl`.
2. **Mood Prediction**:
  * Used mood_prediction_model.pth to classify tracks by mood.
  * Mapped numerical predictions back to mood labels with label_encoder.pkl.
3. **Evaluation**:
  * Generated:
    - Classification Report: Precision, recall, and F1-score for each mood.
    - Confusion Matrix: Visualized mood classification accuracy.
4. **Interactive Playlist Function**:
  * Allows users to:
    - Select a mood from predictions.
    - Randomly generate playlists of up to 10 songs.
    - View playlist details, including track name, artist, and album.
5. **Visualization**:
* Created a mood distribution chart showing the number of tracks per mood.


---

## Future Considerations
1. **Web/App Integration**: The next logical step - to develop a web or mobile interface to make playlist generation user-friendly.
2. **Save Playlist**: Modify the function to save the generated playlist as a .csv file for future use
3. **Playlist Length**: Customize the number of songs in the playlist by modifying the random selection logic.
4. **Dynamic Mood Tracking**: Incorporate real-time user feedback to refine playlists dynamically.
5. **Expand Dataset**: Integrate additional datasets to include a broader range of tracks and genres.
6. **Collaborative Playlists**: Allow users to share mood-based playlists with friends.
7. **Genre Customization**: Add options for users to filter playlists by specific genres.
8. **Language Support**: Expand mood classification to include songs in multiple languages.

## Summary

Mood.fm combines machine learning and audio feature analysis to create mood-based playlists. Unlike traditional platforms, it focuses on audio attributes like valence and energy to classify moods. When Spotify restricted API access, we adapted by utilizing publicly available datasets while preserving the core structure of our original Spotify-based model. This transition allowed us to build on the existing framework and maintain our project *vision*.

# References/Acknowledgments:

Link to our [GoogleDrive](https://drive.google.com/file/d/1asQ54xgKQVuRjvSKScjQSjFgfoTC6d3k/view?usp=drivesdk). Google Drive has been shared with the professor.

### General Reources:

1. [OpenAI. Chatgpt](https://chatgpt.com/?model=gpt-4o-mini): Assisted with documentation and code generation/formatting.
2. [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset): Kaggle Dataset

### Spotify-Specific Resrouces:

1. [Spotify Web API](https://developer.spotify.com/documentation/web-api): Previously used before API access was limited
2. [Audio Features Endpoint](https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-audio-features): Spotify Web API reference for audio features. 
3. [Spotify Accessibility Guidelines](https://developer.spotify.com/documentation/accessibility): Accessibility documentation for Spotify developers.
4. [Spotify Design Guidelines](https://developer.spotify.com/documentation/design): Design guidelines for Spotify's developer tools.
5. [Spotify Developer Documentation](https://developer.spotify.com/): Main developer documentation.
6. [Spotify Research](https://research.atspotify.com/): Spotify's main research page.
7. [Spotify Engineering](https://engineering.atspotify.com/): Spotify's engineering blog.
8. [AI Playlist Expanding (Spotify Newsroom)](https://newsroom.spotify.com/2024-09-24/ai-playlist-expanding-usa-canada-ireland-new-zealand/): **Note**: While we were working on Mood.fm, Spotify began rolling out their AI-generated playlist recommendations in September 2024, aiming for a similar goal.

### Research Papers and Publications:

1. [Automatic Music Playlist Generation via Simulation-Based Reinforcement Learning](https://research.atspotify.com/2023/07/automatic-music-playlist-generation-via-simulation-based-reinforcement-learning/): Spotify Research, July 2023.  
2. [Shifting Consumption Towards Diverse Content via Reinforcement Learning](https://research.atspotify.com/2021/03/shifting-consumption-towards-diverse-content-via-reinforcement-learning/): Spotify Research, March 2021.
3. [How Spotify Uses ML to Create the Future of Personalization](https://engineering.atspotify.com/2021/12/how-spotify-uses-ml-to-create-the-future-of-personalization/): Spotify Engineering, December 2021.  
4. [Socially Motivated Music Recommendation](https://research.atspotify.com/2024/06/socially-motivated-music-recommendation/): Spotify Research, June 2024.  
5. [Socially Motivated Music Recommendation (Publication)](https://research.atspotify.com/publications/socially-motivated-music-recommendation/): Spotify Research.
6. [Global Music Streaming Data Reveals Robust Diurnal and Seasonal Patterns of Affective Preference](https://research.atspotify.com/publications/global-music-streaming-data-reveals-robust-diurnal-and-seasonal-patterns-of-affective-preference/): Spotify Research.
7. [Robust Diurnal and Seasonal Patterns of Affective Preference](https://www.nature.com/articles/s41562-018-0508-z?proof=true): Nature Publication.
8. [Shifting Affective Preferences and Patterns](https://dl.acm.org/doi/full/10.1145/3535101): ACM Digital Library.
9. [ICWSM Article on Music and Mood Analysis](https://ojs.aaai.org/index.php/ICWSM/article/view/31359): AAAI Conference on Web and Social Media.   
  
### GitHub Repositories:

1. [Spotify Deep Learning (GitHub Repository)](https://github.com/ieeecs-ut/spotify-deep-learning): A GitHub repository for deep learning projects related to Spotify.  
2. [GNN-in-RS (GitHub Repository)](https://github.com/wusw14/GNN-in-RS): A GitHub repository for Graph Neural Networks in Recommendation Systems.

### Additional Dataset:

1. [Deam](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music): MediaEval Database for Emotional Analysis 2017. Used in consideration of other datasets, later discarded.


