import numpy as np
import torch
import torch.linalg as la

from .agent import DDQN


def effective_rank(
    features: torch.Tensor,
    srank_threshold: float,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Approximate the effective rank of the current representation.

    Feature rank is defined as the number of singular values greater than some epsilon.
    See the paper for details: https://arxiv.org/pdf/2010.14498.pdf.

    """

    U, S, V = la.svd(features)

    # Effective feature rank is the number of normalized singular values
    # such that their cumulative sum is greater than some epsilon.
    assert (S < 0).sum() == 0, "Singular values cannot be non-negative."
    s_sum = torch.sum(S)

    # Catch case where the regularizer has collapsed the network features
    # This makes the training not crash entirely when rank collapse occurs
    if np.isclose(s_sum.item(), 0.0):
        # Break tie through random selection of two singular values
        indices = torch.randperm(len(S))[:2]
        s_min, s_max = S[indices]
        return torch.zeros(1), s_min, s_max
    else:
        S_normalized = S / s_sum
        S_cum = torch.cumsum(S_normalized, dim=-1)
        # Get the first index where the rank threshold is exceeded
        k = (S_cum > srank_threshold).nonzero()[0]
        return k, S.min(), S.max()
