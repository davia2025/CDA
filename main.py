from finalproject.DownloadData import getData
from finalproject.getGenreClusters import genreClusters
from finalproject.dimReduction import poster_dim_reduction
from finalproject.compare_clusters import comparing_clusters
import numpy as np
import pandas as pd

if __name__ == "__main__":
#Getting clean data
    ratings, posters, movie_ids, genres = getData()
#Getting genre clusters
    genres_with_labels = genreClusters(genres)
#saving to outputs for convenience
    ratings.to_parquet("./data/outputs/ratings.parquet")
    genres_with_labels.to_parquet("./data/outputs/genres.parquet")
    np.save("./data/outputs/posters.npy", posters)
    np.save("./data/outputs/movie_ids.npy", movie_ids)
#reducing dimensions on movie posters
    reduced_posters = poster_dim_reduction(posters)
#saving to outputs for convenience
np.save("./data/outputs/reduced_posters.npy", reduced_posters)
#calculate cluster comparison metrics for clustering by genre vs. clustering by poster
adjusted_rand, v_measure, clustered_movies = comparing_clusters(reduced_posters, ids=movie_ids,  id_col_name="imdb_id", df=genres_with_labels, label_col_name="cluster")
#saving to outputs for convenience
cluster_metrics = pd.DataFrame({"adjustedRand": [adjusted_rand], "homogeneity": [v_measure[0]], "completeness": [v_measure[1]], "v_measure": [v_measure[2]]})
cluster_metrics.to_parquet("./data/outputs/cluster_metrics.parquet")
clustered_movies.to_parquet("./data/outputs/movies_with_clusters.parquet")


