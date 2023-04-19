from functools import partial

from kmeans_pytorch import kmeans, kmeans_predict, pairwise_distance
import torch
import torch.nn as nn
from geomloss import SamplesLoss

wasserstein_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")


MSE = nn.MSELoss()


##  USE COSINE DISTANCE IN HIGH DIMENSIONS

# kmeans = partial(kmeans, distance="cosine")
# kmeans_predict = partial(kmeans_predict, distance="cosine")
# pairwise_distance = partial(pairwise_distance, distance="cosine")

class LossWassersteinFull(nn.Module):
    """Computes the Wasserstein distance between the target and the generated data.

    Args:
        target (torch.Tensor): The target data.
    """

    def __init__(self, target: torch.Tensor, *x):
        super(LossWassersteinFull, self).__init__()
        self.target = target

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the given input tensor.

        Args:
            input (torch.Tensor): The input tensor to compute the loss on.

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        return wasserstein_loss(input, self.target)


class LossKMeans(nn.Module):
    """Precomputes the cluster centers and cluster statistics for the target set.

    Args:
        target (torch.Tensor): The target data.
        n_clusters (int): The number of clusters to use.
    """

    def __init__(self, target: torch.Tensor, n_clusters: int):
        super(LossKMeans, self).__init__()
        prediction, cluster_centers = kmeans(X=target, num_clusters=n_clusters)
        self.cluster_centers = cluster_centers
        prediction = kmeans_predict(target, self.cluster_centers)
        distances = pairwise_distance(target, self.cluster_centers)
        means, covs = cluster_statistics(target, prediction, self.cluster_centers)
        self.filling_target = approx_cluster_filling(distances).detach()
        self.means_target = means
        self.covs_target = covs
        self.prediction = prediction
        self.target = target


class LossMeanCov(nn.Module):
    """Computes loss based on cluster statistics for a given set."""

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the given input tensor.

        Args:
            input (torch.Tensor): The input tensor to compute the loss on.

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        loss_fil = compute_cluster_filling_loss(input, self.cluster_centers, self.filling_target)
        prediction = kmeans_predict(input, self.cluster_centers)
        means, covs = cluster_statistics(input, prediction, self.cluster_centers)
        loss_stat = MSE(means, self.means_target) + MSE(covs, self.covs_target)
        return loss_fil + loss_stat


class LossWasserstein(nn.Module):
    """Computes combined Wasserstein loss in clusters."""

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes the loss for the given input tensor.

        Args:
            input (torch.Tensor): The input tensor to compute the loss on.

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        loss_fil = compute_cluster_filling_loss(input, self.cluster_centers, self.filling_target)
        ## get cluster association
        prediction = kmeans_predict(input, self.cluster_centers)
        ## for each cluster, compute wasserstein distance
        loss_med = torch.tensor(0.)
        for cluster in torch.unique(prediction):
            loss_med += wasserstein_loss(input[prediction == cluster], self.target[self.prediction == cluster])
        return loss_fil + loss_med


def compute_cluster_filling_loss(input: torch.Tensor, cluster_centers: torch.Tensor, filling_target: torch.Tensor, print_filling: bool = False) -> torch.Tensor:
    """Computes the loss based on the cluster filling.
    Args:
    input (torch.Tensor): The input data tensor.
    cluster_centers (torch.Tensor): The tensor containing the cluster centers.
    filling_target (torch.Tensor): The target filling tensor.
    print_filling (bool, optional): Whether to print the filling. Defaults to False.

    Returns:
        torch.Tensor: The computed loss tensor.
    """
    distances = pairwise_distance(input, cluster_centers)
    #   distances = distances / self.mean_distances.repeat(input.shape[0], 1)  ### ?!?!?
    filling = approx_cluster_filling(distances)
    if print_filling:
        print("filling")
        print("target", filling_target)
        print("gen'ted", filling)
    filling = 1 - filling / filling_target
    loss_fil = torch.mean(filling * filling)
    return loss_fil


class LossKMeansMeanCov(LossKMeans, LossMeanCov):
    """ cluster loss based on cluster statistics """
    pass


class LossKMeansWasserstein(LossKMeans, LossWasserstein):
    """ cluster loss based on Wasserstein distance in clusters """
    pass


def approx_cluster_filling(distances):
    """ computes an approximation for the number of datapoints within each cluster.
    This is done by applying the formula

    .. math::
    n_c = \sum_{i=1}^N \frac{\exp(-d_{i,c})}{\sum_{k=1}^K \exp(-d_{i,k})}

    d_i,c - distance of data point i to cluster center c
    c - cluster id
    K - number of clusters

    Args:
    distances (torch.Tensor): The tensor containing the pairwise distances between data points and cluster centers.

    Returns:
    torch.Tensor: The computed filling tensor.

     """
    exp_d = torch.exp(-distances) + 1e-16  # add small epsilon to omit nans
    exp_d = exp_d / exp_d.sum(dim=1, keepdims=True)  ## -> range 0-1
    filling = exp_d.sum(dim=0)
    filling = filling / filling.sum()  # renormalize by total to obtain relative filling
    return filling


def distribution_statistics(data):
    """ computes the distribution statistics of given data for a cluster centered on 0
    Args:
    data (torch.Tensor): The tensor containing the input data.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: A tuple containing the computed mean and covariance tensors.
    """
    mean = torch.mean(data, dim=0)
    cov = torch.cov(data.T)
    return mean, cov


def cluster_statistics(data, clusters, cluster_centers):
    """ computes the distribution statistics for all clusters identified a given data

    Args:
    data (torch.Tensor): The tensor containing the input data.
    clusters (torch.Tensor): The tensor containing the predicted cluster associations for each data point.
    cluster_centers (torch.Tensor): The tensor containing the cluster center positions.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: A tuple containing the computed mean and covariance tensors.
    """
    means, covs = zip(*[distribution_statistics(data[clusters == c] - center)
                        for c, center in enumerate(cluster_centers)])
    return torch.stack(means), torch.stack(covs)
