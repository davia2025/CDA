import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

def optimalPCA(X, var_explained, dimension):
    pca = PCA(n_components=var_explained, random_state=6740)
    data_transformed =  pca.fit_transform(X)
    v = pca.explained_variance_ratio_
    explained_variance = sum(v)
    num_components = pca.n_components_
    print(f"% of {dimension} variance explained by PCA with {num_components} components: {explained_variance}.")
    return data_transformed

def optimalIsomap(X, dimension: str):
    neighbors = np.arange(5,30,5)
    components = np.arange(2,20,2)
    errors = []
    min_error = 10000000000
    best_params = ( 0,0 )
    for neighbor in neighbors:
        for component in components:
            iso = Isomap(n_components= component, n_neighbors= neighbor)
            iso.fit_transform(X)
            error = iso.reconstruction_error()
            errors.append(error)
            if  (min_error-error)/min_error > 0.5:
                min_error = error
                best_params = (component, neighbor)
    best_isomap = Isomap(n_components= best_params[0], n_neighbors= best_params[1])
    X_transformed = best_isomap.fit_transform(X)
    print(f"Optimal {dimension} Isomap: {best_params[0]} components, {best_params[1]} neighbors, {min_error} reconstruction error.")
    return X_transformed

def compare_trustworthiness(*args, raw_data, labels: list):
    score = [0.0]
    best_dr = ""
    best_index = 0
    for i, arg in enumerate(args):
        s = trustworthiness(raw_data, arg)
        if s > max(score):
            best_dr = labels[i]
            best_index = i
        score.append(s)
    print(f"{best_dr} has the highest trustworthiness: {score[best_index+1]}")
    return args[best_index]

def poster_dim_reduction(data):
    s = StandardScaler()
    X = s.fit_transform(data)
    pca = optimalPCA(X, var_explained=0.6, dimension="Poster")
    isomap = optimalIsomap(X, dimension="Poster")
    final_reduction = compare_trustworthiness(pca, isomap, raw_data = X, labels=["PCA", "ISOMAP"])
    return final_reduction
