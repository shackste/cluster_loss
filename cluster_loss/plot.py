import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import scipy
import corner
import umap


class DistributionPlotter:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=2, perplexity=30, random_state=42):
        """
        Initializes the Dimension reduction algorithms with specified parameters.
        :param n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
        :param min_dist: The minimum distance apart that points are allowed to be in the low-dimensional representation.
        :param n_components: The dimension of the space to embed into.
        :param random_state: The random seed used to initialize the algorithm.
        """
        self.umapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)

        self.markers = ['o', '+', '^', 'x' 'D', 'v', '*', 'p', 'h']  # List of marker styles


    def fit(self, target_set):
        """
        Fits the UMAP model on the target set to learn the high-dimensional to low-dimensional mapping.
        :param target_set: The high-dimensional data to fit the UMAP model on.
        """
        self.umapper.fit(target_set)

    def transform(self, data_set):
        """
        Transforms the given data set into the 2D space learned during the fitting process.
        :param data_set: The high-dimensional data to be transformed into the 2D space.
        :return: The 2D embedding of the input data set.
        """
        return self.umapper.transform(data_set)

    def plot_data(self, *datasets, density_contours=True, **kwargs):
        embeddings = [self.transform(d) for d in datasets]
        self.plot_transformed_data(*embeddings, density_contours=density_contours, **kwargs)


    def plot_transformed_data(self, *embeddings, labels=None, density_contours=False, color_by_density=False, color_by_dataset=False, title=None):
        """
        Plots the 2D embeddings of the given data sets.
        :param datasets: Variable number of data sets to be transformed and plotted.
        :param labels: Optional labels for the data sets. Should match the number of data sets if provided.
        """
        if labels and len(embeddings) != len(labels):
            raise ValueError("Number of data sets and labels must match")

        plt.figure(figsize=(10, 8))

        handles = []
        for i, embedding in enumerate(embeddings):
            label = labels[i] if labels else f"Data Set {i+1}"

            if density_contours:
                sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], levels=5, color='C'+str(i), linewidths=1, alpha=0.7, label=label)
                handles.append(mpatches.Patch(color='C'+str(i), label=label))

            if color_by_density:
                # Compute densities
                xy = np.vstack([embedding[:, 0], embedding[:, 1]])
                z = scipy.stats.gaussian_kde(xy)(xy)
                # Color by density
                plt.scatter(embedding[:, 0], embedding[:, 1], c=z, label=label, cmap='viridis', s=50, alpha=0.5, marker=self.markers[i])
            elif color_by_dataset:
                plt.scatter(embedding[:, 0], embedding[:, 1], label=label, alpha=0.5, marker=self.markers[i])



#            plt.scatter(embedding[:, 0], embedding[:, 1], label=label, alpha=0.5)
        if handles:
            plt.legend(handles=handles)
        else:
            plt.legend()
        plt.title(title)
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.show()

    def plot_corner(self, *datasets, labels=None, title=None):
        """
        Plots separate corner plots for each of the given data sets in different colors.
        :param datasets: Variable number of data sets for which to create separate corner plots.
        :param labels: Optional labels for the data sets. Should match the number of data sets if provided.
        """

        # Define a list of colors for the plots
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']

       # Initialize a figure
        fig = plt.figure(figsize=(10, 8))

        for i, data_set in enumerate(datasets):
            # Use the corner plot function with overplotting for multiple data sets
            corner.corner(data_set, color=colors[i % len(colors)], hist_bin_factor=2, fig=plt.gcf())

        # Add a legend to differentiate between data sets
        handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], linewidth=3, linestyle='none', marker='o') for i in range(len(datasets))]
        if not labels:
            labels = [f"Dataset {i+1}" for i in range(len(datasets))]
        plt.legend(handles, labels, loc='upper right', frameon=False, bbox_to_anchor=(1.0, 2.0))

        fig.suptitle(title)
        plt.show()


    def plot_tsne_sets(self, X1, X2, labels=["Dataset 1", "Dataset 2"], marker1="X", marker2="+", title=None):
        """ plot tsne for two datasets, using a common transformation"""
        # Concatenate the datasets
        X_combined = np.vstack((X1, X2))

        # Apply t-SNE to the combined dataset
        X_combined_tsne = self.tsne.fit_transform(X_combined)

        # Split the transformed data back into two datasets
        X1_tsne = X_combined_tsne[:len(X1)]
        X2_tsne = X_combined_tsne[len(X1):]

        # Plot
        plt.figure(figsize=(10, 8))
        plt.title(title)
        plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], label=labels[0], marker=marker1)
        plt.scatter(X2_tsne[:, 0], X2_tsne[:, 1], label=labels[1], marker=marker2)
        plt.legend()


    def plot_tsne_sets(self, *datasets, labels=None, markers=None, title=None):
        """Plot t-SNE for an arbitrary number of datasets, using a common transformation"""
        # Ensure labels and markers are provided for each dataset, use defaults if not
        if labels is None:
            labels = [f"Dataset {i+1}" for i in range(len(datasets))]

        # Concatenate the datasets
        X_combined = np.vstack(datasets)

        # Apply t-SNE to the combined dataset
        X_combined_tsne = self.tsne.fit_transform(X_combined)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.title(title)

        start_idx = 0
        for i, X in enumerate(datasets):
            # Determine the end index for the current dataset
            end_idx = start_idx + len(X)
            
            # Extract the transformed data for the current dataset
            X_tsne = X_combined_tsne[start_idx:end_idx]
            
            # Plot the current dataset
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], label=labels[i], marker=self.markers[i])

            # Update the start index for the next dataset
            start_idx = end_idx

        plt.legend()


