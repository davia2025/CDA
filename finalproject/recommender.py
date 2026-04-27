import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def build_user_item_matrix(ratings: pd.DataFrame):
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

def train_test_split_ratings(ratings, test_size=0.2, random_state=6740):
    return train_test_split(ratings, test_size=test_size, random_state=random_state)

def train_svd(user_item_matrix, n_components=50):
    global_mean = np.nanmean(user_item_matrix.values)

    filled_matrix = user_item_matrix.fillna(global_mean)

    svd = TruncatedSVD(n_components=n_components, random_state=6740)
    latent_matrix = svd.fit_transform(filled_matrix)
    item_matrix = svd.components_

    return svd, latent_matrix, item_matrix, global_mean

def predict_svd(user_id, movie_id, user_item_matrix, svd, latent_matrix, item_matrix):
    try:
        user_idx = user_item_matrix.index.get_loc(user_id)
        item_idx = user_item_matrix.columns.get_loc(movie_id)

        user_vec = latent_matrix[user_idx]
        item_vec = item_matrix[:, item_idx]

        return np.dot(user_vec, item_vec)
    except:
        return np.nan

def evaluate_svd_only(test_df, user_item_matrix, svd, latent_matrix, item_matrix):
    preds, truths = [], []

    for _, row in test_df.iterrows():
        pred = predict_svd(
            row["userId"], row["movieId"],
            user_item_matrix, svd,
            latent_matrix, item_matrix
        )
        if not np.isnan(pred):
            preds.append(pred)
            truths.append(row["rating"])

    rmse = np.sqrt(mean_squared_error(truths, preds))
    mae = mean_absolute_error(truths, preds)

    return rmse, mae

def build_hybrid_dataset(ratings_df, user_item_matrix, svd, latent_matrix, item_matrix, visual_features):
    rows = []
    dropped = 0

    for _, row in ratings_df.iterrows():
        user_id = row["userId"]
        movie_id = row["movieId"]
        true_rating = row["rating"]

        svd_pred = predict_svd(
            user_id, movie_id,
            user_item_matrix,
            svd,
            latent_matrix,
            item_matrix
        )

        if movie_id not in visual_features or np.isnan(svd_pred):
            dropped += 1
            continue

        visual_vec = visual_features[movie_id]
        rows.append(np.hstack([svd_pred, visual_vec, true_rating]))

    data = np.array(rows)
    X = data[:, :-1]
    y = data[:, -1]

    coverage = len(rows) / (len(rows) + dropped)

    return X, y, coverage

def train_regression_models(X_train, y_train):
    models = {}

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    models["linear"] = lin

    best_rmse = float("inf")
    best_model = None
    best_alpha = None

    for alpha in [0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)

        preds = ridge.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, preds))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = ridge
            best_alpha = alpha

    models["ridge"] = best_model
    print(f"Best Ridge alpha: {best_alpha}")

    return models

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        results[name] = {"RMSE": rmse, "MAE": mae}

        print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return results

def run_recommendation_pipeline(ratings, movie_ids, visual_embeddings):

    visual_features = {
        movie_ids[i]: visual_embeddings[i]
        for i in range(len(movie_ids))
    }

    train_df, test_df = train_test_split_ratings(ratings)
    train_matrix = build_user_item_matrix(train_df)

    best_svd = None
    best_rmse = float("inf")

    for k in [20, 50, 100]:
        svd, user_latent, item_latent, _ = train_svd(train_matrix, n_components=k)

        rmse, _ = evaluate_svd_only(test_df, train_matrix, svd, user_latent, item_latent)

        if rmse < best_rmse:
            best_rmse = rmse
            best_svd = (svd, user_latent, item_latent, k)

    svd, user_latent, item_latent, best_k = best_svd
    print(f"Best SVD components: {best_k}")

    svd_rmse, svd_mae = evaluate_svd_only(test_df, train_matrix, svd, user_latent, item_latent)

    X_train, y_train, train_cov = build_hybrid_dataset(
        train_df, train_matrix, svd, user_latent, item_latent, visual_features
    )

    X_test, y_test, test_cov = build_hybrid_dataset(
        test_df, train_matrix, svd, user_latent, item_latent, visual_features
    )

    print(f"Train coverage: {train_cov:.2%}")
    print(f"Test coverage: {test_cov:.2%}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = train_regression_models(X_train, y_train)

    hybrid_results = evaluate_models(models, X_test, y_test)

    return {
        "svd_baseline": {"RMSE": svd_rmse, "MAE": svd_mae},
        "hybrid": hybrid_results
    }

def plot_rmse_comparison(results):
    data = pd.DataFrame({
        "Model": ["SVD Baseline", "Hybrid (Linear)", "Hybrid (Ridge)"],
        "RMSE": [
            results["svd_baseline"]["RMSE"],
            results["hybrid"]["linear"]["RMSE"],
            results["hybrid"]["ridge"]["RMSE"]
        ]
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(data=data, x="Model", y="RMSE")

    for i, row in data.iterrows():
        plt.text(i, row["RMSE"], f"{row['RMSE']:.3f}",
                 ha='center', va='bottom')

    plt.title("RMSE Comparison of Models")
    plt.tight_layout()
    plt.savefig("./data/outputs/rmse_comparison_pretty.png", dpi=300)
    plt.show()