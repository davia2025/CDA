from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.cluster import KMeans
import pandas as pd

def compare_dim_reductions(labels_true, labels_pred):
    adjusted_rand = adjusted_rand_score(labels_true, labels_pred)
    v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)
    return {"adjusted_rand_score": adjusted_rand, "v_measure": v_measure}

def comparing_clusters(X, ids, id_col_name, df, label_col_name):
    kmeans = KMeans(n_clusters=25)
    kmeans.fit_predict(X)
    poster_labels = kmeans.labels_
    poster_df = pd.DataFrame({f"{id_col_name}": ids, "posterLabel": poster_labels})
    merged_df = pd.merge(poster_df, df, on=id_col_name)
    comparisons = compare_dim_reductions(merged_df[label_col_name], merged_df["posterLabel"])
    print(f"Adjusted Rand Score: {comparisons["adjusted_rand_score"]} \n Homogeneity: {comparisons["v_measure"][0]} \n Completeness: {comparisons["v_measure"][1]} \n V-Measure: {comparisons["v_measure"][2]}")
    return comparisons["adjusted_rand_score"], comparisons["v_measure"], merged_df


