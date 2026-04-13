import pandas as pd
from finalproject.dimReduction import optimalPCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

def optimalCluster(X: np.array, dimension: str):
    clusters = [5,10,15,20,25,30,35,40,45,50]
    wcss = []
    min_wcss= 100000000000
    best_k = 0
    for k in clusters:
        kmeans = KMeans(n_clusters=k, random_state=6740)
        kmeans.fit(X)
        kmeans.predict(X)
        wcss.append(kmeans.inertia_)
        if  (min_wcss-kmeans.inertia_)/min_wcss > 0.5:
            min_wcss = kmeans.inertia_
            best_k = k
    best_kmeans = KMeans(n_clusters=best_k, random_state=6740)
    best_kmeans.fit(X)
    best_kmeans.predict(X)
    labels = best_kmeans.labels_
    print(f"Optimal Number of Clusters by {dimension}: {best_k}")
    plt.plot(clusters, wcss)
    plt.axvline(best_k, color="red")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within-Cluster Sum of Squares")
    plt.title(f"Optimal Number of Clusters - {dimension}")
    plt.show()
    return labels




def genreClusters(genre: pd.DataFrame):
    genres = genre.iloc[:, 3:].copy()
#PCA to reduce components for clustering
    genres_transformed = optimalPCA(genres, 0.9, "Genre")
#KMeans clustering
    labels = optimalCluster(genres_transformed, "Genre")
    final_genres = genre.copy()
    final_genres["cluster"] = labels
    return final_genres




