"""UMAP + t-SNE embeddings with KMeans / DBSCAN clustering overlays."""

import json

import matplotlib.pyplot as plt
import numpy as np


def render_embeddings(pred_denorm, targets, patient_ids, cardio_utils, embeddings_dir, prefix):
    """Render four scatter plots (UMAP/t-SNE × KMeans/DBSCAN) and save embedding JSON."""
    try:
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import umap
    except Exception as e:
        print(f"Skipping embeddings - missing dependency: {e}")
        return

    if pred_denorm is None or targets is None:
        print("Skipping embeddings - predictions or targets are None")
        return

    valid_mask = ~(np.isnan(pred_denorm).any(axis=1) | np.isnan(targets).any(axis=1))
    if not valid_mask.any():
        print("Skipping embeddings - all data contains NaN")
        return

    valid_predictions = pred_denorm[valid_mask]
    valid_targets = targets[valid_mask]

    if len(valid_predictions) < 5:
        print("Skipping embeddings - insufficient valid data points")
        return

    combined_data = np.hstack([valid_predictions, valid_targets])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    print("Computing UMAP embedding...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_reducer.fit_transform(scaled_data)

    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled_data) - 1))
    tsne_embedding = tsne.fit_transform(scaled_data)

    print("Performing binary clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)

    cluster_means = []
    for cluster_id in [0, 1]:
        cluster_mask = (kmeans_labels == cluster_id)
        if cluster_mask.any():
            cluster_target_mean = valid_targets[cluster_mask].mean()
            cluster_means.append((cluster_id, cluster_target_mean))

    cluster_means.sort(key=lambda x: x[1], reverse=True)
    positive_cluster = cluster_means[0][0] if cluster_means else 0
    binary_labels = np.where(kmeans_labels == positive_cluster, 0, 1)

    dbscan = DBSCAN(eps=0.3, min_samples=3)
    dbscan_raw_labels = dbscan.fit_predict(scaled_data)
    dbscan_labels = np.where(dbscan_raw_labels == -1, 1, dbscan_raw_labels % 2)

    colors = ["#2E8B57", "#DC143C"]
    labels = ["Positive", "Negative"]

    plot_specs = [
        (umap_embedding, binary_labels, "UMAP with Binary Clustering", "UMAP 1", "UMAP 2",
         f"{prefix}_umap_kmeans_embedding"),
        (umap_embedding, dbscan_labels, "UMAP with DBSCAN Clustering", "UMAP 1", "UMAP 2",
         f"{prefix}_umap_dbscan_embedding"),
        (tsne_embedding, binary_labels, "t-SNE with Binary Clustering", "t-SNE 1", "t-SNE 2",
         f"{prefix}_tsne_kmeans_embedding"),
        (tsne_embedding, dbscan_labels, "t-SNE with DBSCAN Clustering", "t-SNE 1", "t-SNE 2",
         f"{prefix}_tsne_dbscan_embedding"),
    ]

    for embedding, label_arr, title, xlabel, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for idx, label in enumerate(labels):
            mask = (label_arr == idx)
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=colors[idx], label=label, alpha=0.7, s=50)
        ax.set_title(f"{prefix.title()}: {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        cardio_utils.save_figure(fig, filename, subdir="embeddings")
        plt.close(fig)

    embedding_data = {
        "umap_embedding": umap_embedding.tolist(),
        "tsne_embedding": tsne_embedding.tolist(),
        "binary_labels": binary_labels.tolist(),
        "binary_dbscan_labels": dbscan_labels.tolist(),
        "patient_ids": patient_ids if isinstance(patient_ids, list) else list(patient_ids),
    }

    with open(embeddings_dir / f"{prefix}_embedding_data.json", "w") as f:
        json.dump(embedding_data, f, indent=2)
