import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the movie dataset
df = pd.read_csv("movie_metadata_sample.csv")

# Show basic info
print("Original dataset:")
print(df.head())
print("\nSummary:")
print(df.describe())

# Drop non-numeric or identifier columns (optional based on use case)
df = df.drop(['movie_title', 'director_name', 'actor_1_name', 'actor_2_name', 
              'actor_3_name', 'genres', 'plot_keywords', 'language', 
              'country', 'content_rating'], axis=1, errors='ignore')

# Handle missing values by filling them with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Final feature list for clustering
features = df.select_dtypes(include=[np.number])
print("\nUsing these features for clustering:")
print(features.columns)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# =============================
# Step 1: Elbow Method to choose K
# =============================
# Step 1: Elbow Method to choose K 
'''
Elbow method is the technique use to choose to determine the optimal number of
cluster(k) in Kmean clustering
In K-Means, we must manually choose how many clusters (k) we want.
But how do we decide the best value for k? That's where the Elbow Method helps 
 How Does It Work?
Run K-Means clustering on the dataset for different values of k (for example, from 1 to 10).

Calculate the total "inertia" (also called WCSS – Within-Cluster Sum of Squares) for each k.

Inertia measures how spread out the clusters are.

Lower inertia is better, but after a certain point, reducing inertia becomes less useful.

Plot a graph of k vs. inertia.

Look for the point on the graph where the inertia drops sharply and then starts to level off.
This point looks like an "elbow" in the graph – that’s your optimal k.

Inertia
  |
  |       ●
  |     ●
  |   ●
  | ●
  |  \
  |   \__
  |      \_____
  +--------------------
        1  2  3  4  5  6 ... k
The "elbow" is at k=3, where the curve bends and the gain from adding more clusters slows down.


'''
n_samples = X_scaled.shape[0]
K_range = range(1, min(11, n_samples + 1))  # k must be <= number of samples

inertia = []
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)


# Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method - Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# =============================
# Step 2: Fit KMeans with Optimal k
# =============================
optimal_k = 3  # Choose based on elbow graph
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nClustered data sample:")
print(df[['Cluster']].value_counts())

# =============================
# Step 3: PCA for 2D visualization
# =============================
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)

df['PCA1'] = reduced[:, 0]
df['PCA2'] = reduced[:, 1]

# Plotting clusters in 2D
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title("Movie Clusters Visualized (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# =============================
# Step 4: Save the clustered data
# =============================
df.to_csv("movie_clustered_output.csv", index=False)
print("\nClustered data saved as 'movie_clustered_output.csv'")

# =============================
# Step 3: User Defines Number of Clusters
# =============================
k = int(input("👉 Enter the number of clusters to create (e.g., 3): "))
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# =============================
# Step 4: Show PCA Graph of Clusters
# =============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title("🎬 Clusters of Movies (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

# =============================
# Step 5: USER GIVES NEW MOVIE DATA
# =============================
print("\n🎯 Now enter details for a new movie:")

# Ask user for same features as dataset
user_data = {}
for col in features.columns:
    val = float(input(f"Enter {col}: "))
    user_data[col] = val

# Convert to DataFrame and scale it using SAME scaler
user_df = pd.DataFrame([user_data])
user_scaled = scaler.transform(user_df)

# Predict which cluster it belongs to
user_cluster = kmeans.predict(user_scaled)[0]
print(f"\n🧠 Based on clustering, your movie belongs to Cluster #{user_cluster}")

# =============================
# Step 6: Optional – Show Where It Lies in PCA Graph
# =============================
user_pca = pca.transform(user_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.scatter(user_pca[0, 0], user_pca[0, 1], color='black', marker='X', s=200, label='Your Movie')
plt.title("📍 Your Movie Placement in Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.show()
