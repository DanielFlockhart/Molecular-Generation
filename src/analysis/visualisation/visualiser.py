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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns  # Import Seaborn for custom color palettes

class Visualiser:
    def __init__(self,data):
        self.load_data(data)

    def load_data(self,data):
        self.data, self.labels,self.names = self.get_dataset(data)
        self.n_samples, self.n_features = self.data.shape

    def get_dataset(self,dataset):
        '''
        Get every vector from csv file in column "vector"
        '''
        vectors = []
        labels = []
        names = []
        
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
                names.append(row['ID'])
                tqdm_instance.update(1)  # Update progress bar
        tqdm_instance.close()  # Close tqdm after the loop

        return np.array(vectors), labels,names
    
    def tSNE(self,perplexity=25,n_components=2):
        # Initialize t-SNE object with desired parameters
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)  # Reduce perplexity to a suitable value

        # Create a tqdm object to display progress bar
        tqdm_instance = tqdm(total=self.data.shape[0], desc="Performing t-SNE",bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green')

        # Perform t-SNE dimensionality reduction
        X_tsne = np.zeros((self.n_samples, n_components))  # Placeholder for t-SNE results
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
    
    def cluster_once_kmeans(self,clusters):
        print(format_title("Clustering"))
        clusters = KMeans(n_clusters=clusters, random_state=42)
        cluster_labels = clusters.fit_predict(self.data)
        return cluster_labels

    def cluster_kmeans(self, max_clusters):
        print(format_title("Clustering - Using Elbow Method"))
        
        # Calculate the variance for different numbers of clusters
        distortions = []
        for clusters in tqdm(range(1, max_clusters + 1),bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'):
            kmeans = KMeans(n_clusters=clusters, random_state=42)
            kmeans.fit(self.data)
            distortions.append(kmeans.inertia_)
        
        # Plot the elbow curve
        plt.plot(range(1, max_clusters + 1), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion (Variance)')
        plt.title('Elbow Method for Optimal Clusters')
        plt.grid()
        plt.show()

        # Calculate percentage change in distortions
        percentage_change = [100 * ((distortions[i] - distortions[i-1]) / distortions[i-1]) for i in range(1, len(distortions))]

        # Find the elbow point as the index with the maximum percentage change
        elbow_point_index = np.argmax(percentage_change)
        optimal_clusters = elbow_point_index + 1

        print("Optimal number of clusters:", optimal_clusters)

        # Perform KMeans clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.data)
        
        return cluster_labels


    def cluster_agglomerative(self,clusters):
        print(format_title("Agglomerative Clustering"))
        cluster_model = AgglomerativeClustering(n_clusters=clusters)
        cluster_labels = cluster_model.fit_predict(self.data)
        return cluster_labels

    def plot(self,X_model, cluster_labels,perp_label="0",specific_cluster=None,dimension=2):
        print(format_title("Plotting Data"))

        # Create a DataFrame with the t-SNE results and labels
        columns = ["Component 1", "Component 2"]
        if dimension == 3:
            columns.append("Component 3")
        df = pd.DataFrame(X_model, columns=columns)
        df["Label"] = self.labels
        df["Cluster"] = cluster_labels
        df["Name"] = self.names

        
        if specific_cluster is not None:
            # Filter DataFrame for the specific clusteraa
            df = df[df["Cluster"] == specific_cluster]
        temp_df = df[df["Cluster"] == 10]
        # Print every smile in specific cluster here
        print(temp_df["Label"].values)
            
        colours = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray', 'olive']
        if dimension == 3:
            fig = px.scatter_3d(df, x="Component 1", y="Component 2",z="Component 3", color="Cluster", hover_data=["Label","Name"],color_continuous_scale=colours, title=f"t-SNE Plot of Dataset with {perp_label} Perplexity and {specific_cluster} Clusters in {dimension}D")
        else:
            fig = px.scatter(df, x="Component 1", y="Component 2", color="Cluster", hover_data=["Label","Name"],color_continuous_scale=colours, title=f"t-SNE Plot of Dataset with {perp_label} Perplexity and {specific_cluster} Clusters in {dimension}D")

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

    def load_html(self,name):
        '''
        Load HTML File of Graph automatically
        '''
        return open(f"{file_constants.VISUALISATIONS_FOLDER}\\graphs\\{name}.html")
    
    def load_html_specific(self,model_type,clustering_type,clusters,specific):
        pass

if __name__ == "__main__":
    dataset = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db1\inputs.csv"

    visualiser = Visualiser(dataset)
    perps = [35]#[5,15,25,35,45]
    clusters = [4000]#[2,3,6,10,20]
    for perp in perps:
        X_tsne,tsne = visualiser.tSNE(perp)
        for cluster in clusters:
            print(f"Perplexity: {perp} Clusters: {cluster}")
            cluster_labels_kmeans = visualiser.cluster_once_kmeans(cluster)
            fig_tsne = visualiser.plot(X_tsne,cluster_labels_kmeans,perp_label=perp)
            visualiser.save_plot(fig_tsne,clusters,"tsne", clustering_type="kmeans")


    #visualiser.save_plot(fig_tsne,clusters,"tsne",clustering_type="kmeans",specific=0)
    #visualiser.save_plot(fig_tsne,clusters,"tsne",clustering_type="kmeans",specific=1)
    #visualiser.save_plot(fig_tsne,clusters,"tsne",clustering_type="kmeans",specific=5)

    #X_pca,pca = visualiser.PCA()
    #fig_pca = visualiser.plot(X_pca,cluster_labels_kmeans)
    #visualiser.save_plot(fig_pca,clusters,"pca",clustering_type="kmeans")


    # Agglomerative Clustering Takes a long time to run

    #cluster_labels_agglomerative = visualiser.cluster_agglomerative(clusters=44)

    #fig_tsne = visualiser.plot(X_tsne,cluster_labels_agglomerative)
    #fig_pca = visualiser.plot(X_pca,cluster_labels_agglomerative)

    #visualiser.save_plot(fig_tsne,clusters,"tsne",clustering_type="agglomerative")
    #visualiser.save_plot(fig_pca,clusters,"pca",clustering_type="agglomerative")

    


class Latent:
    def __init__(self,values):
        self.data = values

    def tSNE(self,perplexity=25,n_components=2):
        # Initialize t-SNE object with desired parameters
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)  # Reduce perplexity to a suitable value

        # Create a tqdm object to display progress bar
        tqdm_instance = tqdm(total=self.data.shape[0], desc="Performing t-SNE",bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green')

        # Perform t-SNE dimensionality reduction
        X_tsne = np.zeros((self.n_samples, n_components))  # Placeholder for t-SNE results
        batch_size = 240
        for i in range(0, self.n_samples, batch_size):
            current_batch = self.data[i:i + batch_size]
            current_batch_size = current_batch.shape[0]
            X_tsne[i:i + current_batch_size] = tsne.fit_transform(current_batch)
            tqdm_instance.update(current_batch_size)  # Update progress bar

        tqdm_instance.close()  # Close tqdm after the loop
        return X_tsne,tsne

    
    def cluster_kmeans(self,clusters):
        print(format_title("Clustering"))
        clusters = KMeans(n_clusters=clusters, random_state=42)
        cluster_labels = clusters.fit_predict(self.data)
        return cluster_labels


    def plot(self,X_model, cluster_labels,perp_label="0",specific_cluster=None,dimension=2):
        print(format_title("Plotting Data"))

        # Create a DataFrame with the t-SNE results and labels
        columns = ["Component 1", "Component 2"]
        if dimension == 3:
            columns.append("Component 3")
        df = pd.DataFrame(X_model, columns=columns)
        df["Label"] = self.labels
        df["Cluster"] = cluster_labels
        df["Name"] = self.names

        
        if specific_cluster is not None:
            # Filter DataFrame for the specific cluster
            df = df[df["Cluster"] == specific_cluster]
        colours = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray', 'olive']
        if dimension == 3:
            fig = px.scatter_3d(df, x="Component 1", y="Component 2",z="Component 3", color="Cluster", hover_data=["Label","Name"],color_continuous_scale=colours, title=f"t-SNE Plot of Dataset with {perp_label} Perplexity and {specific_cluster} Clusters in {dimension}D")
        else:
            fig = px.scatter(df, x="Component 1", y="Component 2", color="Cluster", hover_data=["Label","Name"],color_continuous_scale=colours, title=f"t-SNE Plot of Dataset with {perp_label} Perplexity and {specific_cluster} Clusters in {dimension}D")

        # Show the plot
        fig.show()
        return fig