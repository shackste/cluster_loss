## this file contains the definition of distances to be used in the other files
## this ensures consistency in the use of different distance metrics
from functools import partial

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

from kmeans_pytorch import kmeans, kmeans_predict, pairwise_cosine
from kmeans_pytorch import pairwise_distance as pairwise_distance_orig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kmeans_orig = partial(kmeans, device=device)
kmeans_predict_orig = partial(kmeans_predict, device=device)
pairwise_distance_orig = partial(pairwise_distance_orig, device=device)
pairwise_cosine_orig = partial(pairwise_cosine, device=device)



class DistanceMetric:
  """ container for computations that involve distance metrics, for consistent usage """
  # Globale Parameter
  use_euclid_distance = False ## force use of euclid distance
  use_cosine_distance = False ## force use of cosine distance
  max_dim_euclid = 5          ## threshold to change to cosine distance

  mode = "Euclid"
  kmeans = kmeans_orig
  kmeans_predict = kmeans_predict_orig
  pairwise_distance = pairwise_distance_orig

  wasserstein_distance = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")

  
  @classmethod
  def set_distance_metrics(cls, dimensionality, threshold=max_dim_euclid):
      """ sets all internal distance metrics to Euclidean distance (low dimension) or cosine similarity (high dimension), depending on dimensionality 
  
      Args:
          dimensionality (int): number of dimensions in the considered space
  
      Kwargs:
          threshold (int): if dimensionality is lower than threshold, use euclid, else, use cosine. Default: 5 (set in max_dim_euclid)
  
      Note:
          you can set either use_euclid_distance or use_cosine_distance to True, to force this mode. If both are True, Euclid is used.
      """      
      if cls.use_euclid_distance or (dimensionality <= threshold and not cls.use_cosine_distance):
          print(f"Dimensionality: {dimensionality}<={threshold} -> use Euclid distance")
          mode = "Euclid"
          cls.kmeans = kmeans_orig
          cls.kmeans_predict = kmeans_predict_orig
          cls.pairwise_distance = pairwise_distance_orig
      else:
          print(f"Dimensionality: {dimensionality}>{threshold} -> use Cosine distance")
          mode = "Cosine"
          # fastest way to compute cosine distance is to compute euclid distance on normalized vectors
          cls.kmeans = lambda *args, X=None, **kwargs: kmeans_orig(*args, X=F.normalize(X, dim=1), **kwargs)
          cls.kmeans_predict = lambda a, b: kmeans_predict_orig(F.normalize(a), F.normalize(b))
          cls.pairwise_distance = lambda a, b: pairwise_distance_orig(F.normalize(a), F.normalize(b))
          ### this computation uses the same trick, but normalization is done repeatedly
#          cls.kmeans = partial(kmeans_orig, distance="cosine")
#          cls.kmeans_predict = partial(kmeans_predict_orig, distance="cosine")
#          cls.pairwise_distance = pairwise_cosine_orig
