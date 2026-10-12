import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
import numpy as np
import math


def compute_cos_similarity_matrix(image_features1, image_features2):
    image_features1 = F.normalize(image_features1, dim=-1)
    image_features2 = F.normalize(image_features2, dim=-1)
    return image_features1 @ image_features2.T
def compute_bhattacharyya_distance_matrix(image_features1, image_features2, eps=1e-6):
    """
    Compute the Bhattacharyya distance matrix between all sample pairs in the batch

    Args:
        mu1, sigma1_sq: [B, D]
        mu2, sigma2_sq: [B, D]

    Returns:
        distance_matrix: [B, B]
    """

    mu1, sigma1_sq = image_features1
    mu2, sigma2_sq = image_features2
    mu1 = F.normalize(mu1, dim=-1)
    mu2 = F.normalize(mu2, dim=-1)
    B, D = mu1.shape


    mu1_exp = mu1.unsqueeze(1)  # [B, 1, D]
    mu2_exp = mu2.unsqueeze(0)  # [1, B, D]
    sigma1_sq_exp = sigma1_sq.unsqueeze(1)  # [B, 1, D]
    sigma2_sq_exp = sigma2_sq.unsqueeze(0)  # [1, B, D]

    sigma_avg_sq = 0.5 * (sigma1_sq_exp + sigma2_sq_exp)  # [B, B, D]

    diff = mu1_exp - mu2_exp  # [B, B, D]
    term1 = (diff * diff) / (sigma_avg_sq + eps)  # [B, B, D]
    term1 = term1.sum(dim=2)  # [B, B]

    log_det_avg = torch.log(sigma_avg_sq + eps).sum(dim=2)  # [B, B]
    log_det1 = torch.log(sigma1_sq + eps).sum(dim=1).unsqueeze(1)  # [B, 1]
    log_det2 = torch.log(sigma2_sq + eps).sum(dim=1).unsqueeze(0)  # [1, B]

    term2 = log_det_avg - 0.5 * (log_det1 + log_det2)  # [B, B]

    bd_matrix = 0.125 * term1 + 0.5 * term2  # [B, B]
    similarity_matrix = torch.exp(-bd_matrix / D)
    return similarity_matrix

def min_max_norm(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)
def confident_loss(image_features1, image_features2, labels):
    mu1, sigma1_sq = image_features1
    mu2, sigma2_sq = image_features2
    mu1 = F.normalize(mu1, dim=-1)
    mu2 = F.normalize(mu2, dim=-1)
    B, D = mu1.shape

    mu1_exp = mu1.unsqueeze(1)  # [B, 1, D]
    mu2_exp = mu2.unsqueeze(0)  # [1, B, D]


    diff = mu1_exp - mu2_exp  # [B, B, D]
    term1 = diff * diff
    term1 = term1.sum(dim=2)  # [B, B]
    similarity_matrix = torch.exp(-term1)
    mask = ~torch.eye(B, dtype=torch.bool, device=similarity_matrix.device)  # (b, b)

    similarity_no_diag = similarity_matrix[mask].view(B, B - 1)
    similarity_diag = similarity_matrix[~mask]
    neg_similarity, _ = torch.topk(similarity_no_diag, k=1, dim=1, largest=True)
    uncertainty = torch.exp(-similarity_diag / torch.mean(neg_similarity, dim=1))
    avg_sigma = (torch.mean(sigma1_sq, dim=1) + torch.mean(sigma2_sq, dim=1)) / 2

    var_loss = uncertainty.mean(dim=-1)
    uncertainty_norm = F.normalize(uncertainty, dim=-1)
    sigma_norm = F.normalize(avg_sigma, dim=-1)

    cos_sim = F.cosine_similarity(uncertainty_norm, sigma_norm, dim=-1)
    return 1-cos_sim,var_loss*0.5



class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu', metric='bd'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device
        if metric == 'cos':
            self.prob_metric_func = compute_cos_similarity_matrix
        elif metric == 'bd':
            self.prob_metric_func = compute_bhattacharyya_distance_matrix


    def forward(self, image_features1, image_features2, logit_scale,state='mean'):

        logits_per_image1 = logit_scale * self.prob_metric_func(image_features1, image_features2)

        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss_pro = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2
        if state=='std':
            loss_rank,var_loss = confident_loss(image_features1, image_features2, labels)
            return loss_pro, loss_rank * 2.4, var_loss

        else:
            loss = loss_pro
            return loss




