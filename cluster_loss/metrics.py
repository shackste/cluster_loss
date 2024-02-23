from functools import partial

import torch
from kmeans_pytorch import pairwise_distance
from geomloss import SamplesLoss
from pytorch_fid.fid_score import calculate_frechet_distance

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: USE COSINE DISTANCE IN HIGH DIMENSIONS
# pairwise_distance = partial(pairwise_cosine)
pairwise_distance = partial(pairwise_distance, device=device)


wasserstein_distance = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")

def calculate_fid(real_features, generated_features):
    # Calculate the mean and covariance of the real features
    real_mean = np.mean(real_features.numpy(), axis=0)
    real_covariance = np.cov(real_features, rowvar=False)

    # Calculate the mean and covariance of the generated features
    generated_mean = np.mean(generated_features.numpy(), axis=0)
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
    filling = 1 - filling / filling_target
    loss_fil = torch.mean(filling * filling)
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
