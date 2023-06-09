from functools import partial

from kmeans_pytorch import kmeans, kmeans_predict, pairwise_distance
import torch
import torch.nn as nn
from geomloss import SamplesLoss

from cluster_loss.metrics import compute_cluster_filling_mse, approx_cluster_filling, cluster_statistics, calculate_fid

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wasserstein_distance = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")

MSE = nn.MSELoss()

# TODO: USE COSINE DISTANCE IN HIGH DIMENSIONS

# kmeans = partial(kmeans, distance="cosine")
# kmeans_predict = partial(kmeans_predict, distance="cosine")
# pairwise_distance = partial(pairwise_cosine)

kmeans = partial(kmeans, device=device)
kmeans_predict = partial(kmeans_predict, device=device)
pairwise_distance = partial(pairwise_distance, device=device)


class LossWassersteinFull(nn.Module):
    """ Computes the Wasserstein distance between the target and the generated data.

    Args:
        target (torch.Tensor): The target data.
    """

    def __init__(self, target: torch.Tensor, *x):
        super(LossWassersteinFull, self).__init__()
        self.target = target.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor to compute the loss on.

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        # Calculate the sizes of both arrays
        size_x = len(x)
        size_target = len(self.target)

        # Find the minimum size and resize both arrays accordingly
        min_size = min(size_x, size_target)
        x_resized = x[:min_size]
        target_resized = self.target[:min_size]

        # Compute the Wasserstein distance with same-sized arrays
        return wasserstein_distance(x_resized, target_resized)


class LossKMeans(nn.Module):
    """Precomputes the cluster centers and cluster statistics for the target set.

    Args:
        target (torch.Tensor): The target data.
        n_clusters (int): The number of clusters to use.
    """

    def __init__(self, target: torch.Tensor, n_clusters: int):
        super(LossKMeans, self).__init__()
        with torch.no_grad():
            prediction, cluster_centers = kmeans(X=target, num_clusters=n_clusters, device=target.device)
            prediction = kmeans_predict(target, cluster_centers)
            distances = pairwise_distance(target, cluster_centers)
            means, covs = cluster_statistics(target, prediction.to(target.device), cluster_centers.to(target.device))
        self.cluster_centers = cluster_centers.detach()
        self.filling_target = approx_cluster_filling(distances).detach()
        self.means_target = means.detach()
        self.covs_target = covs.detach()
        self.prediction = prediction.detach()
        self.target = target.detach()


class LossMeanCov(nn.Module):
    """Computes loss based on cluster statistics for a given set."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor to compute the loss on.

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        loss_fil = compute_cluster_filling_mse(x, self.cluster_centers.to(x.device), self.filling_target.to(x.device))
        prediction = kmeans_predict(x, self.cluster_centers.to(x.device))
        means, covs = cluster_statistics(x, prediction.to(x.device), self.cluster_centers.to(x.device))
        loss_stat = MSE(means, self.means_target.to(x.device)) + MSE(covs, self.covs_target.to(x.device))  # TODO: Instead of MSE, use FID
        return loss_fil + loss_stat


class LossWasserstein(nn.Module):
    """Computes combined Wasserstein loss in clusters."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor to compute the loss on.

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        loss_fil = compute_cluster_filling_mse(x, self.cluster_centers.to(x.device), self.filling_target.to(x.device))
        # get cluster association
        prediction = kmeans_predict(x, self.cluster_centers.to(x.device)).detach()
        # for each cluster, compute wasserstein distance
        loss_med = torch.tensor(0.).to(device)
        for cluster in torch.unique(prediction):
            in_cluster = prediction == cluster
            if not in_cluster.any():
                continue
            # Calculate the sizes of both arrays
            size_x = len(x[in_cluster])
            size_target = len(self.target[self.prediction == cluster])

            # Find the minimum size and resize both arrays accordingly
            min_size = min(size_x, size_target)
            x_resized = x[in_cluster][:min_size]
            target_resized = self.target[self.prediction == cluster][:min_size]

            # Compute the Wasserstein distance with same-sized arrays
            loss_med += wasserstein_distance(x_resized.to(device), target_resized.to(device))

        return loss_fil + loss_med


class LossFID(nn.Module):
    def __init__(self):
        super(LossFID, self).__init__()

    def forward(self, real_features, generated_features):
        fid = calculate_fid(real_features, generated_features)
        return fid


class LossKMeansMeanCov(LossKMeans, LossMeanCov):
    """ cluster loss based on cluster statistics """
    pass


class LossKMeansWasserstein(LossKMeans, LossWasserstein):
    """ cluster loss based on Wasserstein distance in clusters """
    pass
