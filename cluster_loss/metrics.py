from functools import partial

import numpy as np
import torch
from kmeans_pytorch import kmeans, kmeans_predict, pairwise_distance
from geomloss import SamplesLoss
from pytorch_fid.fid_score import calculate_frechet_distance

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: USE COSINE DISTANCE IN HIGH DIMENSIONS
# pairwise_distance = partial(pairwise_cosine)
pairwise_distance = partial(pairwise_distance, device=device)


wasserstein_distance = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")


class ClusterMetrics:
    @torch.no_grad()
    def __init__(self, target: torch.Tensor, n_clusters: int):
        self.n_clusters = n_clusters
        prediction, self.cluster_centers = kmeans(X=target, num_clusters=n_clusters, device=target.device)
        distances = pairwise_distance(target, self.cluster_centers)
        self.filling_target = torch.bincount(prediction)
        self.filling_target /= self.filling_target.sum()
#        self.mean_distances = torch.tensor([torch.mean(distances[prediction == c,c])
#                                             for c in range(self.n_clusters)])
        self.cluster_distances_target = self.compute_cluster_distances(distances,
                                                                       prediction,
                                                                       torch.ones(n_clusters))
        self.cluster_stddevs_target = self.compute_cluster_standard_deviation(distances,
                                                prediction,
                                                torch.ones(n_clusters),
                                                self.cluster_distances_target)


    def cluster_metrics(self, data):
        """ Compute the cluster metrics for the given data.
        The K cluster centers are fitted to the target set used for initialization of the parent ClusterMetrics object.

        Args:
            data (torch.Tensor): NxD-Tensor containing N datapoints with D dimensions

        Returns:
            error (torch.Tensor): 1-Tensor containing the cluster error score (Eq. 1)
            distances (torch.Tensor): K-Tensor containing the cluster distance score for each cluster (Eq. 2)
            stddev (torch.Tensor): K-Tensor containing the cluster standard deviation score for each cluster (Eq. 3)
        """
        error = self.cluster_error(data)
        distances, stddevs = self.cluster_distances_and_standard_deviations(data)
        return error, distances, stddevs

    @torch.no_grad()
    def cluster_error(self, data):
        filling = self.cluster_filling(data)
        return cluster_error(filling, self.filling_target)

    @torch.no_grad()
    def cluster_filling(self, data):
        prediction = self.predict_cluster(data)
        filling = torch.bincount(prediction)
        filling /= filling.sum()
        return filling

    @torch.no_grad()
    def cluster_distances(self, data):
        """ compute the cluster distance for each cluster """
        distances = pairwise_distance(data, self.cluster_centers)
        prediction = self.predict_cluster_from_distances(distances)
        return self.compute_cluster_distances(distances,
                                              prediction,
                                              self.cluster_distances_target)

    @torch.no_grad()
    def compute_cluster_distances(self, distances, prediction, target):
        cluster_distances = []
        for c in range(self.n_clusters):
            d = distances[prediction==c, c]
            cd = cluster_distance(d, target[c])
            cluster_distances.append(cd)
        return torch.tensor(cluster_distances)

    def cluster_distances_and_standard_deviations(self, data):
        """ compute the cluster standard deviation for each cluster """
        distances = pairwise_distance(data, self.cluster_centers)
        prediction = self.predict_cluster_from_distances(distances)
        dd = self.compute_cluster_distances(distances,
                                            prediction,
                                            torch.ones(n_clusters))
        cluster_distances = dd / self.cluster_distances_target
        cluster_stddevs = self.compute_cluster_standard_deviation(distances,
                                                                  prediction,
                                                                  self.cluster_stddevs_target,
                                                                  dd)
        return cluster_distances, cluster_stddevs


    @torch.no_grad()
    def compute_cluster_standard_deviation(self, distances, prediction, target, dd):
        cluster_stddevs = []
        for c in range(self.n_clusters):
            d = distances[prediction==c, c]
            csd = cluster_standard_deviation(d, target[c], dd[c])
            cluster_stddevs.append(csd)
        return torch.tensor(cluster_stddevs)




    @torch.no_grad()
    def predict_cluster(self, data):
        return kmeans_predict(data, self.cluster_centers)

    @torch.no_grad()
    def predict_cluster_from_distances(self, distances):
        return torch.argmin(distances, dim=1)

def cluster_distance(d, d_target):
    """
    Computes the cluster distance (Hackstein et al. 2023, Eq. 2)

    Args:
        d (torch.Tensor): distances to neighbouring cluster center
        d_target (float): computed cluster distance for target set

    Returns:
        torch.Tensor: the computed cluster distance value
    """
    return torch.sqrt(torch.mean(d*d)) / d_target

def cluster_standard_deviation(d, sigma_target, dd):
    """
    Computes the cluster standard deviation (Hackstein et al. 2023, Eq. 3)

    Args:
        d (torch.Tensor): distances to neighbouring cluster center
        sigma_target (float): computed cluster standard deviation for target set
        dd (float): non-renormalized cluster distance (d*D)

    Returns:
        torch.Tensor: the computed cluster standard deviation value
    """
    d = d - dd

    return torch.sqrt(torch.mean(d*d)) / sigma_target

def cluster_error(n: torch.Tensor, n_target: torch.Tensor):
    """
    Computes the cluster error (Hackstein et al. 2023)

    Args:
        n (torch.Tensor): number of data points within each cluster
        n_target (torch.Tensor): target number of ...

    Returns:
        torch.Tensor: the computed error
    """
    diff = 1 - n / n_target
    return torch.mean(diff*diff)


def calculate_fid(real_features, generated_features):
    # Calculate the mean and covariance of the real features
    real_features = real_features.cpu().numpy()
    generated_features = generated_features.cpu().numpy()
    real_mean = np.mean(real_features, axis=0)
    real_covariance = np.cov(real_features, rowvar=False)

    # Calculate the mean and covariance of the generated features
    generated_mean = np.mean(generated_features, axis=0)
    generated_covariance = np.cov(generated_features, rowvar=False)

    return calculate_frechet_distance(real_mean, real_covariance, generated_mean, generated_covariance)

def torch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    '''Estimate a covariance matrix given data.
    '''
    # Calculate the mean of the data along the specified axis.
    mean = torch.mean(x, dim=0, keepdim=True) if not rowvar else torch.mean(x, dim=1, keepdim=True)

    # Subtract the mean from the data.
    x = x - mean

    # Calculate the weighted covariance matrix.
    if aweights is not None:
        raise NotImplementedError('aweights is not implemented yet.')
    else:
        factor = 1.0 / (x.size(1) - 1) if not bias else 1.0 / x.size(1)
        cov = factor * torch.matmul(x, x.t())
        if ddof is not None:
            if isinstance(ddof, int):
                cov *= float(x.size(1) - ddof) / float(x.size(1) - 1)
            else:
                raise ValueError('ddof must be an integer.')
    return cov


def compute_cluster_filling_mse(input: torch.Tensor, cluster_centers: torch.Tensor, filling_target: torch.Tensor, print_filling: bool = False) -> torch.Tensor:
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
    #   distances = distances / self.mean_distances.repeat(input.shape[0], 1)  # Todo: normalize by mean distance?
    filling = approx_cluster_filling(distances)
    if print_filling:
        print("filling")
        print("target", filling_target)
        print("gen'ted", filling)
    loss_fil = cluster_error(filling, filling_target)
    return loss_fil

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
    if data.shape[0] == 0: # empty cluster causes nan in torch.mean and torch.cov
        return torch.zeros(data.shape[1]).to(data.device), torch.zeros((data.shape[1], data.shape[1])).to(data.device)
    mean = torch.mean(data, dim=0)
    if data.shape[0] == 1: # single data point causes nan in torch.cov
        return mean, torch.zeros((data.shape[1], data.shape[1])).to(data.device)
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
