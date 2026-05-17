import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


# Create dataset
data = {
    'Posts_Shared': [2, 3, 4, 25, 26],
    'Likes_Received': [10, 12, 11, 80, 85]
}

df = pd.DataFrame(data)

print(df.head())


# Apply DBSCAN
dbscan = DBSCAN(eps=5, min_samples=2)

df['Cluster'] = dbscan.fit_predict(
    df[['Posts_Shared', 'Likes_Received']]
)


# Separate clusters and noise
clusters = df[df['Cluster'] != -1]

noise = df[df['Cluster'] == -1]


# Plot clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    clusters['Posts_Shared'],
    clusters['Likes_Received'],
    c=clusters['Cluster'],
    cmap='viridis',
    label='Clusters',
    s=50
)

plt.scatter(
    noise['Posts_Shared'],
    noise['Likes_Received'],
    c='red',
    marker='x',
    label='Noise (Outliers)',
    s=50
)

plt.title('DBSCAN Clustering')
plt.xlabel('Posts Shared')
plt.ylabel('Likes Received')

plt.legend()
plt.grid(True)

plt.show()


# Print results
print(f"Number of clusters found: {len(set(df['Cluster'])) - (1 if -1 in df['Cluster'].values else 0)}")

print(f"Number of noise points: {len(noise)}")


# Cohesion and Separation
if len(set(df['Cluster'])) > 1:
    
    score = silhouette_score(
        df[['Posts_Shared', 'Likes_Received']],
        df['Cluster']
    )

    print(f"Silhouette Score: {score}")
