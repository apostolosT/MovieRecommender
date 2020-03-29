import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_movies(target_movie,similarity_coefficients, k=4):
    '''Returns list of top-k movies similar to target movie'''
   # Get all coefficients for the target movie
    similar_movies = similarity_coefficients.loc[target_movie]
    
    # Drop the target_movies as we don't want to report the Target Movie as being similar to Target Movie
    similar_movies = similar_movies.drop(target_movie)
    
    # Sort in Descending order. More similar ones should come first
    similar_movies = similar_movies.sort_values(ascending=False)
    
    # Leave only the ones that are positively correlated with target movie
    similar_movies = similar_movies[similar_movies > 0]
    
    # Return top-k results
    return similar_movies.head(k)


def get_rated_movies_by_user(target_user,ratings):
	is_rated = ratings.loc[target_user].notna()
	rated_movies = ratings.loc[target_user][is_rated].index

	return rated_movies

def find_s_ij(movie1,movie2,similarity_matrix):
    return similarity_matrix.loc[movie1,movie2]

def find_s_ij_per_user(target_movie,list_of_rated_movies,similarity_matrix):
    #list_of_rated_movies.remove(target_movie)
    sim_scores=[]
    for i in list_of_rated_movies:
        sim_scores.append(find_s_ij(target_movie,i,similarity_matrix))
    return sim_scores

def average_collaboritive_similarity(list_of_rated_movies):
    return np.average(list_of_rated_movies)
    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 1.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out





