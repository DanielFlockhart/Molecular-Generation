import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans,AgglomerativeClustering
from tqdm import tqdm
import sys,os,csv,ast
sys.path.insert(0, os.path.abspath('..'))
from Constants import ui_constants,file_constants
from ui.terminal_ui import *

class Visualiser:
    def __init__(self,data):
        self.load_data(data)

    def load_data(self,data):
        self.data, self.labels = self.get_dataset(data)
        self.n_samples, self.n_features = self.data.shape

    def get_dataset(self,dataset):
        '''
        Get every vector from csv file in column "vector"
        '''
        vectors = []
        labels = []
        
        # Count the total number of rows in the CSV file
        with open(dataset, newline='') as csvfile:
            num_rows = sum(1 for line in csvfile)
        
        # Reopen the CSV file to reset the iterator
        with open(dataset, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Create a tqdm object to display progress bar
            tqdm_instance = tqdm(total=num_rows, desc="Reading CSV data",bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green')
            
            for row in reader:
                vectors.append(np.array(ast.literal_eval(row['vector'])))
                labels.append(row['SMILES'])
                tqdm_instance.update(1)  # Update progress bar
                
        tqdm_instance.close()  # Close tqdm after the loop

        return np.array(vectors), labels
    
    def tSNE(self,perplexity=49):
        # Initialize t-SNE object with desired parameters
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)  # Reduce perplexity to a suitable value

        # Create a tqdm object to display progress bar
        tqdm_instance = tqdm(total=self.data.shape[0], desc="Performing t-SNE",bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green')

        # Perform t-SNE dimensionality reduction
        X_tsne = np.zeros((self.n_samples, 2))  # Placeholder for t-SNE results
        batch_size = 240
        for i in range(0, self.n_samples, batch_size):
            current_batch = self.data[i:i + batch_size]
            current_batch_size = current_batch.shape[0]
            X_tsne[i:i + current_batch_size] = tsne.fit_transform(current_batch)
            tqdm_instance.update(current_batch_size)  # Update progress bar

        tqdm_instance.close()  # Close tqdm after the loop
        return X_tsne,tsne
    
    def PCA(self, n_components=2):
        # Initialize PCA object with desired parameters
        pca = PCA(n_components=n_components, random_state=42)

        # Create a tqdm object to display progress bar
        tqdm_instance = tqdm(total=self.data.shape[0], desc="Performing PCA", bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green')

        # Perform PCA dimensionality reduction
        X_pca = np.zeros((self.n_samples, n_components))  # Placeholder for PCA results
        batch_size = 240
        for i in range(0, self.n_samples, batch_size):
            current_batch = self.data[i:i + batch_size]
            current_batch_size = current_batch.shape[0]
            X_pca[i:i + current_batch_size] = pca.fit_transform(current_batch)
            tqdm_instance.update(current_batch_size)  # Update progress bar

        tqdm_instance.close()  # Close tqdm after the loop
        return X_pca, pca
    
    def cluster_kmeans(self,clusters):
        print(format_title("Clustering"))
        clusters = KMeans(n_clusters=clusters, random_state=42)
        cluster_labels = clusters.fit_predict(self.data)
        return cluster_labels

    def cluster_agglomerative(self,clusters):
        print(format_title("Agglomerative Clustering"))
        cluster_model = AgglomerativeClustering(n_clusters=clusters)
        cluster_labels = cluster_model.fit_predict(self.data)
        return cluster_labels

    def plot(self,X_model, cluster_labels, specific_cluster=None):
        print(format_title("Plotting Data"))

        # Create a DataFrame with the t-SNE results and labels
        df = pd.DataFrame(X_model, columns=["Component 1", "Component 2"])
        df["Label"] = self.labels
        df["Cluster"] = cluster_labels

        if specific_cluster is not None:
            # Filter DataFrame for the specific cluster
            specific_df = df[df["Cluster"] == specific_cluster]

            # Create an interactive scatter plot with colors for the specific cluster and hover labels using Plotly
            fig = px.scatter(specific_df, x="Component 1", y="Component 2", color="Cluster", hover_data=["Label"])
        else:
            # Create an interactive scatter plot with colors for clusters and hover labels using Plotly
            fig = px.scatter(df, x="Component 1", y="Component 2", color="Cluster", hover_data=["Label"])

        # Show the plot
        fig.show()
        return fig
        

    def save_plot(self,plot,clusters,model_type,clustering_type,specific=None):
        '''
        Save HTML File of Graph automatically
        '''
        if specific is not None:
            plot.write_html(f"{file_constants.VISUALISATIONS_FOLDER}\\graphs\\{model_type}\\{clustering_type}\\{clusters}\\{specific}.html")
        else:
            plot.write_html(f"{file_constants.VISUALISATIONS_FOLDER}\\graphs\\{model_type}\\{clustering_type}\\{clusters}.html")


clusters = 100

if __name__ == "__main__":
    dataset = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Ai-Chem-Intership\data\datasets\db1\inputs.csv"

    visualiser = Visualiser(dataset)
    X_pca,pca = visualiser.PCA()
    X_tsne,tsne = visualiser.tSNE()

    cluster_labels_kmeans = visualiser.cluster_kmeans(clusters)
    fig_tsne = visualiser.plot(X_tsne,cluster_labels_kmeans)
    fig_pca = visualiser.plot(X_pca,cluster_labels_kmeans)

    visualiser.save_plot(fig_tsne,clusters,"tsne",clustering_type="kmeans")
    visualiser.save_plot(fig_pca,clusters,"pca",clustering_type="kmeans")


    # Agglomerative Clustering Takes a long time to run

    cluster_labels_agglomerative = visualiser.cluster_agglomerative(clusters)

    fig_tsne = visualiser.plot(X_tsne,cluster_labels_agglomerative)
    fig_pca = visualiser.plot(X_pca,cluster_labels_agglomerative)

    visualiser.save_plot(fig_tsne,clusters,"tsne",clustering_type="agglomerative")
    visualiser.save_plot(fig_pca,clusters,"pca",clustering_type="agglomerative")

    