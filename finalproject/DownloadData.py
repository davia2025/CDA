import os
import kagglehub
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer



def getData():
    #download latest version of movie poster dataset from Kaggle
    kagglehub.dataset_download("phiitm/movie-posters", output_dir ="../data")

    #download latest version of imdb ratings dataset from Kaggle
    kagglehub.dataset_download("rounakbanik/the-movies-dataset", output_dir ="../data/ratings")
    ratings = pd.read_csv("../data/ratings/ratings.csv")
    rating_movie_ids = list(set(ratings["movieId"]))
    print(f"{len(rating_movie_ids)} movies from ratings dataset.")


    #Store images as numpy array and imdb IDs as list
    movie_id = []
    movies = []
    movie_folder = Path("../data/poster_downloads")
    #shapes = []

    files = sorted(list(movie_folder.glob("*.jpg")))
    print(f"{len(files)} movies from poster dataset.")


    for file in files:
        img = np.array(Image.open(file))
        shape = img.flatten().shape
    #21 posters have different shape from the rest. Removing those for analysis
        if shape == (146328,):
            #shapes.append( img.flatten().shape)
            file_id = int(file.name.split("_")[1].replace(".jpg", ""))
    #only keeping movies with IDs in the ratings file
            if file_id in rating_movie_ids:
                movie_id.append(file_id)
                movies.append(img.flatten())
    movies = np.array(movies)
#only keeping ratings that match movies with posters
    ratings = ratings[ratings.movieId.isin(movie_id)]

    print(f"{len(movie_id)} common movies in ratings and poster datasets.")

#creating one-hot-encoded dataframe of movie genres
    metadata = pd.read_csv("../data/ratings/movies_metadata.csv")
    metadata = metadata[["imdb_id", "title", "genres"]]
#parsing stringified json for genres
    metadata["genres"] = metadata["genres"].apply(ast.literal_eval).apply(lambda x: [g['name'] for g in x])
#one hot encoding for genres - to be used in clustering
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(metadata["genres"])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)
    genres = pd.concat([metadata, genre_df], axis=1)
    genres = genres.dropna(subset=["imdb_id"])
    genres["imdb_id"] = genres["imdb_id"].astype(str).str.replace("tt", "").astype(int)
# only keeping genre encodings that match movies with posters
    genres = genres[genres.imdb_id.isin(movie_id)]
# final join - keeping only movies that have a genre, poster, and ratings
    common_index = [i for i, v in enumerate(movie_id) if v in list(genres["imdb_id"])]
    movie_id = [movie_id[i] for i in common_index]
    movies = movies[common_index,:]
    print(f"{len(movie_id)} common movies in ratings, poster, and genre datasets. \n These will be used for analysis.")
    return ratings, movies, movie_id, genres


