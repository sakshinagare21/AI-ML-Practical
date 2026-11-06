# Hierarchical Clustering

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load Dataset
data = pd.read_csv("Mall_Customers.csv")

# Step 2: Display Dataset
print("First 5 Records:")
print(data.head())

# Step 3: Dataset Information
print("\nDataset Shape:", data.shape)
print("\nMissing Values:")
print(data.isnull().sum())

print("\nData Types:")
print(data.dtypes)

# Step 4: Remove CustomerID
data = data.drop("CustomerID", axis=1)

# Step 5: Select Features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Step 6: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Create Dendrogram
plt.figure(figsize=(10, 6))

linked = linkage(X_scaled, method="ward")

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Step 8: Apply Hierarchical Clustering
model = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

y_hc = model.fit_predict(X_scaled)

# Step 9: Add Cluster Labels
data["Cluster"] = y_hc

print("\nClustered Data:")
print(data.head())

# Step 10: Evaluate Model
score = silhouette_score(X_scaled, y_hc)

print("\nSilhouette Score:", score)

print("\nNumber of Customers in Each Cluster:")
print(data["Cluster"].value_counts())

# Step 11: Visualize Clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=y_hc,
    cmap="rainbow"
)

plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.title("Hierarchical Customer Segmentation")

plt.show()

# Step 12: Analyze Clusters
print("\nCluster Analysis:")
print(data.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean())