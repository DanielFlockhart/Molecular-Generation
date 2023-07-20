import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from get_data import *
from tqdm import tqdm
import sys,os
sys.path.insert(0, os.path.abspath('..'))
from Constants import ui_constants
from ui.terminal_ui import *


# Generate or load your high-dimensional data
# X = ...
data, labels = get_dataset()
n_samples, n_features = data.shape
perplexity = 49  # Set perplexity to a suitable value
# Initialize t-SNE object with desired parameters
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)  # Reduce perplexity to a suitable value

# Create a tqdm object to display progress bar
tqdm_instance = tqdm(total=data.shape[0], desc="Performing t-SNE",bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green')

# Perform t-SNE dimensionality reduction
X_tsne = np.zeros((n_samples, 2))  # Placeholder for t-SNE results
batch_size = 240
for i in range(0, n_samples, batch_size):
    current_batch = data[i:i + batch_size]
    current_batch_size = current_batch.shape[0]
    X_tsne[i:i + current_batch_size] = tsne.fit_transform(current_batch)
    tqdm_instance.update(current_batch_size)  # Update progress bar

tqdm_instance.close()  # Close tqdm after the loop


# ADD IN ELBOW METHOD HERE AND ADD TQDM TO IT

# Apply clustering algorithm to assign cluster labels
print(format_title("Clustering"))
kmeans = KMeans(n_clusters=500, random_state=42)
cluster_labels = kmeans.fit_predict(data)

# Create a DataFrame with the t-SNE results and labels
df = pd.DataFrame(X_tsne, columns=["Component 1", "Component 2"])
df["Label"] = labels
df["Cluster"] = cluster_labels

# Create an interactive scatter plot with colors for clusters and hover labels using Plotly
fig = px.scatter(df, x="Component 1", y="Component 2", color="Cluster", hover_data=["Label"])

# Show the plot
fig.show()