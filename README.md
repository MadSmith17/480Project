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
   * `numpy`
   * `pandas`
   * `scikit-learn`
   * `seaborn`
   * `torch`
   * `torchvision`
   * `tqdm`

## Project Structure

The project directory includes the following:
* NewData/notebooks/: Main project folder containing all resources and scripts
   * `data_preprocessing.ipynb`: Notebook for cleaning and preprocessing the dataset
   * `features_dataset.csv`: Dataset with extracted features for training the model.
   * `metadata_dataset.csv`; Metadata associated with the dataset
   * `mood_prediction_model.pth`: Pre-trained neural network model for mood classification.
   * `NN.ipynb`: Notebook for training the neural network.
   * `testing_model.ipynb`: Notebook for testing and evaluating the model.
   * `resources/dataset.csv`: Original raw dataset from Kaggle.
 
## Steps to Run the Project

### Step 1: Clone the repository to your local machine

### Step 2: Preprocess the Data (Optional)
  1. Open the `data_preprocessing.ipynb` notebook in Jupyter Notebook, VSCode or any compatible IDE.
  2. Run each cell to:
      * Clean the raw dataset (`resources/dataset.csv`).
      * Generate `features_dataset.csv` (for training) and `metadata_dataset.csv` (for additional metadata).
      * Outputs:
         * `features_dataset.csv`: Extracted features, saved in the `notebooks` directory.
         * `metadata_dataset.csv`: Metadata for future analysis, saved in the save directory.

***Note**: This step is optional as these files already exist in this folder.*

### Step 3: Train the Neural Network (Optional)
   1. Open the `NN.ipynb` notebook.
   2. Execute the cells to train the neural network using `features_dataset.csv`.
     * This will overwrite `mood_prediction_model.pth` with a newly trained model.
   3. The training process:
      * Uses extracted features as input to the model.
      * Outputs a trained model (mood_prediction_model.pth) saved in the notebooks directory.
   4. Output:
      * A new model file (mood_prediction_model.pth) will overwrite the existing one.

### Step 4: Test and Evaluate the Model
   1. Open the `testing_model.ipynb` notebook.
   2. Run all cells to
      * Load the pre-trained model `mood_prediction_model.pth`.
      * Evaluate the model on test data from `features_dataset.csv`.
   3. Generated Outputs:
      * Classification Report: Displays model performance metrics (precision, recall, F1-score, etc.).
      * Confusion Matrices: Matrix visualizing the model's predictions vs. actual labels
      * Accuracy and Loss graphs: Plots tracking the model's training/testing performance

 ***Output**: All results, including evaluation metrics, should be saved within the notebook outputs or specific folders.*

 ### Step 5: Generate Playlists
 1. Run the playlist generation function:
    * The playlist generation function is included in `testing_model.ipynb`.
    * It uses the column `predicted_mood` from the dataset to categorize songs into mood-based playlists.
    * The model predicts a mood label for each song based on extracted features.
    * Songs are grouped by mood labels
 2. Steps to Generate a playlist:
    * When you run the function, you'll see a list of mood options  (1, `happy`, 2. `sad`, 3. `calm`, and 4. `energetic`).
    * Enter the number corresponding to the mood you's like to generate a playlist for.
 3. What Happens Next:
    * The function filters songs from the dataset based on the selected mood
    * It randomly selects songs to create a playlist and displays the details, including:
      * **Track Name**: The name of the song.
      * ** Artists**: The song's artist(s).
      * ** Album Name**: The album the song belongs to.
  4. **Example Output**: After selecting a mood, the program generates outputs like the following:

 ### Integration Details
 * The function uses the `predicted_mood` colunb from the processed dataset (`features_dataset.csv`) for catgorization.
 * Randomization enssures varies results from each time you generate a playlist.

### Additional Notes:
* If you'd like to modify the function (i.e. to change the number of songs per playlist), update the logic in the script!
* Ensure the dataset and required models are properly laoded before running the playlist generation function.


# References/Acknowledgments:
[1] [OpenAI. Chatgpt](https://chatgpt.com/?model=gpt-4o-mini): Gpt-4 model. Accessed: 2024

[2] [Spotify Web API](https://developer.spotify.com/documentation/web-api) to access and manage Spotify's music data (before it accessibility became limited)

[3] [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) Kaggle. Retrieved [11/28/2024]

[4] [Deam](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music): MediaEval Database for Emotional Analysis 2017. Dataset retrieved from Kaggle, as the official website was inaccessible. Anna Aljanaki, Mohammad Soleymani, and Frans Wiering. 

