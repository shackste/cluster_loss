from functools import partial

from kmeans_pytorch import kmeans, kmeans_predict, pairwise_distance
import torch
import torch.nn as nn
from geomloss import SamplesLoss

from cluster_loss.metrics import compute_cluster_filling_mse, approx_cluster_filling, cluster_statistics, calculate_fid

wasserstein_distance = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")

MSE = nn.MSELoss()


# TODO: USE COSINE DISTANCE IN HIGH DIMENSIONS

# kmeans = partial(kmeans, distance="cosine")
# kmeans_predict = partial(kmeans_predict, distance="cosine")
# pairwise_distance = partial(pairwise_distance, distance="cosine")

class LossWassersteinFull(nn.Module):
    """ Computes the Wasserstein distance between the target and the generated data.

    Args:
        target (torch.Tensor): The target data.
    """

    def __init__(self, target: torch.Tensor, *x):
        super(LossWassersteinFull, self).__init__()
        self.target = target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor to compute the loss on.

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        # TODO: ensure x and target are of same size
        return wasserstein_distance(x, self.target)


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
        loss_fil = compute_cluster_filling_mse(x, self.cluster_centers, self.filling_target)
        prediction = kmeans_predict(x, self.cluster_centers)
        means, covs = cluster_statistics(x, prediction.to(x.device), self.cluster_centers.to(x.device))
        loss_stat = MSE(means, self.means_target) + MSE(covs, self.covs_target)  # TODO: Instead of MSE, use FID
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
        loss_fil = compute_cluster_filling_mse(x, self.cluster_centers, self.filling_target)
        # get cluster association
        prediction = kmeans_predict(x, self.cluster_centers)
        # for each cluster, compute wasserstein distance
        loss_med = torch.tensor(0.)
        for cluster in torch.unique(prediction):
            loss_med += wasserstein_distance(x[prediction == cluster], self.target[self.prediction == cluster])
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
