import torch
import numpy as np
from tqdm import tqdm
import gc
from rpf_base.trainer_regular import predict
import math
import torch.nn.functional as F
import copy

def evaluate(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True,state='mean'):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader,state=state)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader,state=state)
    
    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0

    if state == 'mean':
        for i in tqdm(range(len(ids_query))):
            ap_tmp, CMC_tmp = eval_query_mean(img_features_query[i], ql[i],
                                             img_features_gallery, gl)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
    else:
        for i in tqdm(range(len(ids_query))):
            ap_tmp, CMC_tmp = eval_query_std((img_features_query[0][i],img_features_query[1][i]), ql[i], img_features_gallery, gl,i)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return CMC[0]

def compute_bhattacharyya_similarity_low_memory(mu1, sigma1_sq, mu2, sigma2_sq, eps=1e-6, gallery_batch_size=1000):

    M, D = mu1.shape
    N = mu2.shape[0]


    similarity = torch.zeros((M, N), device=mu1.device, dtype=mu1.dtype)

    log_det1 = torch.log(sigma1_sq + eps).sum(dim=1)  # [M]

    num_gallery_batches = (N + gallery_batch_size - 1) // gallery_batch_size

    for batch_idx in range(num_gallery_batches):
        start_idx = batch_idx * gallery_batch_size
        end_idx = min((batch_idx + 1) * gallery_batch_size, N)

        mu2_batch = mu2[start_idx:end_idx]  # [B, D]
        sigma2_sq_batch = sigma2_sq[start_idx:end_idx]  # [B, D]
        B = mu2_batch.shape[0]

        mu1_exp = mu1.unsqueeze(1)  # [M, 1, D]
        mu2_batch_exp = mu2_batch.unsqueeze(0)  # [1, B, D]

        diff = mu1_exp - mu2_batch_exp  # [M, B, D]

        sigma_avg = 0.5 * (sigma1_sq.unsqueeze(1) + sigma2_sq_batch.unsqueeze(0))  # [M, B, D]

        term1 = (diff ** 2) / (sigma_avg + eps)  # [M, B, D]
        term1 = 0.125 * term1.sum(dim=2)  # [M, B]

        log_det_avg = torch.log(sigma_avg + eps).sum(dim=2)  # [M, B]
        log_det2_batch = torch.log(sigma2_sq_batch + eps).sum(dim=1)  # [B]

        log_det1_exp = log_det1.unsqueeze(1)  # [M, 1]
        log_det2_exp = log_det2_batch.unsqueeze(0)  # [1, B]

        term2 = 0.5 * (log_det_avg - 0.5 * (log_det1_exp + log_det2_exp))  # [M, B]

        bd = term1 + term2  # [M, B]
        batch_similarity = torch.exp(-bd/D)  # [M, B]

        similarity[:, start_idx:end_idx] = batch_similarity

    return similarity
def compute_bhattacharyya_distance_matrix(mu1, sigma1_sq, mu2, sigma2_sq, eps=1e-6,i=0):

    D = mu1.size(0)
    N = mu2.size(0)

    mu1_exp = mu1.unsqueeze(0).expand(N, D)  # [1, D] -> [N, D]
    sigma1_sq_exp = sigma1_sq.unsqueeze(0).expand(N, D)

    mu2 = mu2
    sigma2_sq = sigma2_sq

    # Σ = 0.5*(Σ1 + Σ2)
    sigma_avg_sq = 0.5 * (sigma1_sq_exp + sigma2_sq)  # [N, D]

    #(μ1 - μ2)^T Σ^{-1} (μ1 - μ2)
    diff = mu1_exp - mu2  # [N, D]
    term1 = (diff * diff) / (sigma_avg_sq + eps)  # [N, D]
    term1 = term1.sum(dim=1)  # [N]

    # 0.5 * log(|Σ| / sqrt(|Σ1||Σ2|))
    log_det_avg = torch.log(sigma_avg_sq + eps).sum(dim=1)  # [N]
    log_det1 = torch.log(sigma1_sq_exp + eps).sum(dim=1)  # [N]
    log_det2 = torch.log(sigma2_sq + eps).sum(dim=1)  # [N]
    term2 = log_det_avg - 0.5 * (log_det1 + log_det2)  # [N]


    bd = 0.125 * term1 + 0.5 * term2  # [N]
    similarity_vector = torch.exp(-bd / D)
    return similarity_vector  # [N]


def eval_query_mean(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp



def eval_query_std(qf, ql, gf, gl,i=0):
    qf_mean,qf_var = qf
    gf_mean,gf_var = gf
    score = compute_bhattacharyya_distance_matrix(qf_mean, qf_var, gf_mean, gf_var,i=i)
    score = score.cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
