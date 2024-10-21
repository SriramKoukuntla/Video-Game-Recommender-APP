import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy import stats
from sklearn.metrics.pairwise import linear_kernel  # for cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up nltk data path
nltk.data.path.append(r'C:\Users\Srira\AppData\Roaming\nltk_data')

# Download necessary nltk data
nltk.download('punkt', download_dir=r'C:\Users\Srira\AppData\Roaming\nltk_data')
nltk.download('stopwords')
nltk.download('wordnet')


# Load dataset
data = pd.read_csv('all_games.csv')

# Remove duplicates
data.drop_duplicates('name', keep='first', inplace=True)
data.reset_index(drop=True, inplace=True)

# Check for null values and remove rows with null 'summary'
null_rows = data[data['summary'].isnull()].index
data.drop(index=null_rows, inplace=True)

# Reset index of proc_data (processed data)
proc_data = data.copy()  # Make a copy for processing
proc_data.reset_index(drop=True, inplace=True)

# Define TF-IDF Vectorizer Object
tfidf = TfidfVectorizer()

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform([str(i) for i in proc_data['summary']])

# Print shape of TF-IDF matrix

# List of feature names
feature_names = tfidf.get_feature_names_out()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reverse mapping of indices and video game titles
indices = pd.Series(proc_data.index, index=proc_data['name'])

# Recommendation function
def recommender_system(title, cosine_sim=cosine_sim):
    # Get index of video game that matches the title
    idx = indices[title]
    
    # Get pairwise similarity scores of all video games with the given title
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort games based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get scores of 10 most similar video games
    sim_scores = sim_scores[1:11]
    
    # Get game indices
    game_indices = [i[0] for i in sim_scores]
    
    # Return top 10 most similar video games
    recs = proc_data['name'].iloc[game_indices]
    
    return recs

# Test run of the recommender system
print("Recommendations for 'Grand Theft Auto V':")
print(recommender_system('Grand Theft Auto V'))