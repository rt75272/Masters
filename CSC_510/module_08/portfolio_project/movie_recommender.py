import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
"""
Movie Recommendation Machine

The user interacts with the program in order to retrieve movie recommendations.

Usage:
    $ python movie_recommender.py
"""
# Initial Setup
movies_file = 'ml-100k/u.item'
ratings_file = 'ml-100k/u.data'
# The movies dataset is read with columns for movieId and title
movies_df = pd.read_csv(movies_file, sep='|', header=None, encoding='latin-1', usecols=[0, 1], names=['movieId', 'title'])
# The ratings dataset is read with columns for userId, movieId, rating, and timestamp
ratings_df = pd.read_csv(ratings_file, sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
# Create a dictionary mapping movieId to movie title for easy reference
movies_dict = dict(zip(movies_df['movieId'], movies_df['title']))
# Create a user-item rating matrix
user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
# Fill missing values with 0 (indicating that a user hasn't rated a movie yet)
user_movie_matrix = user_movie_matrix.fillna(0)
# Calculate cosine similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)
# Convert the movie similarity matrix into a DataFrame for easy access
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

def recommend_movies_for_selected_movie(selected_movie_id, top_n=5):
    """Recommend movies based on the selected movie's similarity score"""
    # Ensure the selected movie exists in the movie similarity matrix
    if selected_movie_id not in movie_similarity_df.columns:
        print(f"Movie ID {selected_movie_id} not found in the movie matrix.")
        return []
    # Get the similarity scores for the selected movie
    similarity_scores = movie_similarity_df[selected_movie_id]
    # Sort the movies based on similarity scores in descending order
    similar_movie_ids = similarity_scores.sort_values(ascending=False).index[1:top_n+1]
    # Convert movie ids to movie titles using the movies dictionary
    recommended_movies = [movies_dict[movie_id] for movie_id in similar_movie_ids]
    return recommended_movies

def main():
    """Main driver function"""
    while True:
        print("\nSelect a movie from the list below (type 'exit' to quit):")
        # This list shows the movieId and movie title
        movie_list = movies_df[['movieId', 'title']].head(50)
        # Print each movie in the movie list with its index number
        for idx, movie in movie_list.iterrows():
            print(f"{idx + 1}. {movie['title']}")
        # Get user input for movie selection
        user_input = input("Enter the number corresponding to the movie you want or type 'exit' to quit: ").strip().lower()
        # If the user enters 'exit', exit the program
        if user_input == 'exit':
            print("Goodbye! Thank you for using the movie recommender.")
            break
        # Handle valid movie selection
        if user_input.isdigit():
            selected_movie_idx = int(user_input) - 1
            # Ensure the selected movie index is valid
            if selected_movie_idx >= 0 and selected_movie_idx < len(movie_list):
                selected_movie_id = movie_list.iloc[selected_movie_idx]['movieId']
                selected_movie_title = movie_list.iloc[selected_movie_idx]['title']
                print(f"\nYou selected: {selected_movie_title}")
                # Recommend movies based on the selected movie
                recommended_movies = recommend_movies_for_selected_movie(selected_movie_id, top_n=5)
                if recommended_movies:
                    print(f"Recommended movies for you based on '{selected_movie_title}':")
                    for movie in recommended_movies:
                        print(movie)
                else:
                    print("No recommendations found.")
            else:
                print("Invalid selection, please choose a number from the list.")
        else:
            print("Invalid input. Please enter a number corresponding to a movie or 'exit' to quit.")

# Big red activation button
if __name__ == '__main__':
    main()