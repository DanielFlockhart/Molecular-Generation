import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from get_data import *
# Generate or load your high-dimensional data
# X = ...
data,labels = get_dataset()
# Initialize t-SNE object with desired parameters
# Initialize t-SNE object with desired parameters
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

# Perform t-SNE dimensionality reduction
X_tsne = tsne.fit_transform(data)

# Apply clustering algorithm to assign cluster labels
kmeans = KMeans(n_clusters=50, random_state=42)
cluster_labels = kmeans.fit_predict(data)


# Create a DataFrame with the t-SNE results and labels
df = pd.DataFrame(X_tsne, columns=["Component 1", "Component 2"])
df["Label"] = labels
df["Cluster"] = cluster_labels

# Create an interactive scatter plot with colors for clusters and hover labels using Plotly
fig = px.scatter(df, x="Component 1", y="Component 2", color="Cluster", hover_data=["Label"])

# Show the plot
fig.show()