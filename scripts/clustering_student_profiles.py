import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import os

def run_clustering():
    # === File paths ===
    profile_path = 'data/interim/profile_cleaned.csv'
    model_dir = 'models'
    figure_dir = 'figures'
    output_path = 'data/interim/profile_with_clusters.csv'

    # === Load the cleaned profile dataset ===
    df = pd.read_csv(profile_path)

    # === Encode categorical features for clustering ===
    category_maps = {
        'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2}
    }
    for col, mapping in category_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1)

    # === Select behavioral features for clustering ===
    features = [
        'Attendance', 'Hours_Studied', 'Sleep_Hours',
        'Tutoring_Sessions', 'Motivation_Level',
        'Parental_Involvement', 'Learning_Disabilities', 'Physical_Activity'
    ]

    # === Standardize the features ===
    scaler = StandardScaler()
    X = df[features]
    X_scaled = scaler.fit_transform(X)

    # === KMeans Clustering ===
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # === Save the model and scaler ===
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans_student_profiles.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_student_profiles.pkl"))
    print("✅ KMeans model and scaler saved to ../models/")

    # === Dimensionality Reduction for Visualization ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    os.makedirs(figure_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title('Student Clusters (via PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'clustering_student_profiles.png'))
    plt.show()

    # === Analyze Cluster Characteristics ===
    cluster_summary = df.groupby('Cluster')[features].mean()
    print("\nCluster Behavioral Profiles:")
    print(cluster_summary)

    # === Optional: Save clustered data ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Clustered data saved to: {output_path}")