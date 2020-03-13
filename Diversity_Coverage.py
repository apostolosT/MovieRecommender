import pandas as pd
import os
import numpy as np

from CollabFilter import filter_data, predict_user_item



if __name__ == '__main__':
    LIST_LENGHT = 10
    data_path = os.getcwd() + '/ml-latest-small/'
    # configure file path
    movies_filename = 'movies.csv'
    ratings_filename = 'ratings.csv'# read data
    df_movies = pd.read_csv(
        os.path.join(data_path, movies_filename))

    df_ratings = pd.read_csv(
        os.path.join(data_path, ratings_filename))
    df_movies.genres = df_movies.genres.str.split(pat = "|")
    df_movies.loc[:, 'category_name'] = df_movies.genres.map(lambda x: x[0])
    # list_of_categories = df_movies.category.unique()
    df_movies.category_name = pd.Categorical(df_movies.category_name)
    df_movies['category'] = df_movies.category_name.cat.codes

    #Cleaning Dataset

    df_ratings_filtered = filter_data(df_ratings, 5, 5)

    n_users = df_ratings_filtered.userId.unique().shape[0]
    n_items = df_ratings_filtered.movieId.unique().shape[0]
    print(str(n_users) + ' users')
    print(str(n_items) + ' items')
    unique_movie_id = df_ratings_filtered.movieId.unique()
    df_ratings_filtered['newId'] = df_ratings_filtered['movieId'].apply(new_movie_id, args=(unique_movie_id,))


    #Calculate user-item ratings matrix R

    item_prediction = predict_user_item(df_ratings_filtered)
    best_movies_index = np.argsort(item_prediction, axis=1)