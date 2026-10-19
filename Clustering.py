# ============================================================
# WEEK 3 TASK: UNSUPERVISED LEARNING AND CLUSTERING ANALYSIS
# Dataset: Mall Customer Segmentation Dataset
# Algorithm: K-Means Clustering
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. CREATE FIGURES FOLDER
# ============================================================

os.makedirs("figures", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Dataset file should be inside the dataset folder
file_path = "dataset/Mall_Customers.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED SUCCESSFULLY ==========")
print(df.head())


# ============================================================
# 3. BASIC DATA EXPLORATION
# ============================================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Remove duplicate rows if any
df = df.drop_duplicates()

# Check missing values
if df.isnull().sum().sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values found and handled.")

# Customer ID is not useful for clustering
df = df.drop(columns=["CustomerID"])


# ============================================================
# 5. SELECT FEATURES FOR CLUSTERING
# ============================================================

# We will use Annual Income and Spending Score
features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

print("\n========== SELECTED FEATURES ==========")
print(X.head())


# ============================================================
# 6. STANDARDIZE THE DATA
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n========== DATA STANDARDIZED ==========")
print(X_scaled[:5])


# ============================================================
# 7. ELBOW METHOD
# ============================================================

wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)


# Plot Elbow Method
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method for Optimal Number of Clusters")

plt.grid(True)

plt.savefig(
    "figures/elbow_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. APPLY K-MEANS CLUSTERING
# ============================================================

# Based on the Elbow Method, we use 5 clusters
optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ============================================================
# 9. DISPLAY CLUSTER COUNTS
# ============================================================

print("\n========== CLUSTER COUNTS ==========")

cluster_counts = df["Cluster"].value_counts().sort_index()

print(cluster_counts)


# ============================================================
# 10. CLUSTER CENTERS
# ============================================================

cluster_centers_scaled = kmeans.cluster_centers_

cluster_centers = scaler.inverse_transform(
    cluster_centers_scaled
)

cluster_centers_df = pd.DataFrame(
    cluster_centers,
    columns=features
)

print("\n========== CLUSTER CENTERS ==========")

print(cluster_centers_df)


# ============================================================
# 11. CLUSTER CHARACTERISTICS
# ============================================================

cluster_summary = df.groupby("Cluster")[features].mean()

print("\n========== CLUSTER CHARACTERISTICS ==========")

print(cluster_summary)


# ============================================================
# 12. SILHOUETTE SCORE
# ============================================================

silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("\n========== SILHOUETTE SCORE ==========")

print(
    f"Silhouette Score: {silhouette:.4f}"
)


# ============================================================
# 13. CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set2",
    s=100
)

plt.title("Customer Segmentation using K-Means Clustering")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.legend(title="Cluster")

plt.grid(True)

plt.savefig(
    "figures/cluster_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. CLUSTER DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

cluster_counts.sort_index().plot(
    kind="bar"
)

plt.title("Number of Customers in Each Cluster")

plt.xlabel("Cluster")

plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.grid(axis="y")

plt.savefig(
    "figures/cluster_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. SAVE PROCESSED DATASET
# ============================================================

df.to_csv(
    "clustered_customer_data.csv",
    index=False
)

print("\n========== PROJECT COMPLETED ==========")

print(
    "\nProcessed dataset saved as: "
    "clustered_customer_data.csv"
)

print(
    "\nGraphs saved inside the 'figures' folder."
)

print(
    "\nFinal cluster summary:"
)

print(cluster_summary)

print(
    f"\nFinal Silhouette Score: {silhouette:.4f}"
)