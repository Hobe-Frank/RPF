import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, state='mean'):
    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for query, reference, ids in bar:

        if scaler:
            with autocast():

                # data (batches) to device
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)

                # Forward pass
                features1, features2 = model(query, reference, state=state)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:

                    loss_pro, rank_loss,var_loss2 = loss_function(features1, features2, model.module.logit_scale.exp(),
                                                        state=state)
                else:
                    loss_pro, rank_loss,var_loss2 = loss_function(features1, features2, model.logit_scale.exp(),state=state)
                if state == 'std':
                    mu1, sigma1_sq = features1
                    mu2, sigma2_sq = features2
                    var_loss = 0.4 * (sigma1_sq.mean(dim=-1) + sigma2_sq.mean(dim=-1)).mean()
                    loss = loss_pro + var_loss + rank_loss + var_loss2
                else:
                    loss = loss_pro

                losses.update(loss.item())

            scaler.scale(loss).backward()

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            monitor = {"contrast_loss": "{:.4f}".format(loss_pro.item()),
                       "var_loss": "{:.4f}".format(var_loss.item()),
                       "var_loss_conf": "{:.4f}".format(var_loss2.item()),
                       "rank_loss": "{:.4f}".format(rank_loss.item()),
                       "loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

def predict_vigor(train_config, model, dataloader, img_type='query', state='mean'):
    model.eval()

    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    with torch.no_grad():
        dummy_output = model(img1=dummy_input, state=state)
    # print(isinstance(dummy_output, tuple),len(dummy_output) == 2)
    # 判断输出是否为 (mu, var) 元组
    if isinstance(dummy_output, tuple) and len(dummy_output) == 2:
        output_shape = dummy_output[0].shape[1:]  # mu 的特征维度
        use_probabilistic = True
    else:
        output_shape = dummy_output.shape[1:]
        use_probabilistic = False

    # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
    total_samples = len(dataloader.dataset)
    img_features_mu = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
    # Assuming each id_current has 4 elements, adjust the dimension for ids
    ids = torch.zeros((total_samples, 4), dtype=torch.long, device=train_config.device)  # 修改为二维张量以匹配ids_current的形状
    if use_probabilistic:
        img_features_var = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
    else:
        img_features_var = None
    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            model_output = model(img1=img, state=state)
            if use_probabilistic:
                mu, var = model_output
                mu = F.normalize(mu, dim=-1)
                img_features_mu[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = mu
                img_features_var[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = var
            else:
                mu = model_output
                mu = F.normalize(mu, dim=-1)
                img_features_mu[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = mu
            # Directly assign the 2D ids_current to the corresponding slice in ids
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    if use_probabilistic:
        return (img_features_mu, img_features_var), ids
    else:
        return img_features_mu, ids
def compute_bhattacharyya_distance_matrix(mu1, sigma1_sq, mu2, sigma2_sq):
    """
    计算批量中所有样本对之间的Bhattacharyya距离矩阵

    Args:
        mu1, sigma1_sq: [B, D]
        mu2, sigma2_sq: [B, D]

    Returns:
        distance_matrix: [B, B]
    """
    B, D = mu1.shape
    eps = 1e-8
    # # 确保方差为正
    # sigma1_sq = sigma1_sq.clamp(min=self.eps)
    # sigma2_sq = sigma2_sq.clamp(min=self.eps)

    # 扩展维度用于广播计算 [B, 1, D] 和 [1, B, D]
    mu1_exp = mu1.unsqueeze(1)  # [B, 1, D]
    mu2_exp = mu2.unsqueeze(0)  # [1, B, D]
    sigma1_sq_exp = sigma1_sq.unsqueeze(1)  # [B, 1, D]
    sigma2_sq_exp = sigma2_sq.unsqueeze(0)  # [1, B, D]

    # 平均协方差: Σ = 0.5 * (Σ1 + Σ2)
    sigma_avg_sq = 0.5 * (sigma1_sq_exp + sigma2_sq_exp)  # [B, B, D]

    # 第一项: (μ1 - μ2)^T Σ^{-1} (μ1 - μ2)
    diff = mu1_exp - mu2_exp  # [B, B, D]
    term1 = (diff * diff) / (sigma_avg_sq + eps)  # [B, B, D]
    term1 = term1.sum(dim=2)  # [B, B]

    # 第二项: 0.5 * log(|Σ| / sqrt(|Σ1||Σ2|))
    # 使用对数求和而不是乘积，避免数值问题
    log_det_avg = torch.log(sigma_avg_sq + eps).sum(dim=2)  # [B, B]
    log_det1 = torch.log(sigma1_sq + eps).sum(dim=1).unsqueeze(1)  # [B, 1]
    log_det2 = torch.log(sigma2_sq + eps).sum(dim=1).unsqueeze(0)  # [1, B]

    term2 = log_det_avg - 0.5 * (log_det1 + log_det2)  # [B, B]

    # 正确的Bhattacharyya距离
    bd_matrix = 0.125 * term1 + 0.5 * term2  # [B, B]

    return bd_matrix


# def predict(train_config, model, dataloader):
#     model.eval()
#     # Get output shape from a dummy input
#     dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
#     output_shape = model(dummy_input).shape[1:]
#
#     # Pre-allocate memory for efficiency (assuming fixed batch size)
#     img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
#     ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
#
#     with torch.no_grad(), autocast():
#         for i, (img, ids_current) in enumerate(tqdm(dataloader)):
#             img = img.to(train_config.device)
#             img_feature = model(img)
#
#             # normalize is calculated in fp32
#             if train_config.normalize_features:
#                 img_feature = F.normalize(img_feature, dim=-1)
#
#             img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
#             ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current
#
#     return img_features, ids

# def predict(train_config, model, dataloader,img_type='query',state='mean'):
#     """
#     推理函数，支持模型输出为 (mu, var) 元组的概率嵌入模型
#
#     Args:
#         train_config: 包含 device, normalize_features 等配置
#         model: 模型，输出为 (mu, var) 或 mu（兼容旧模型）
#         dataloader: 数据加载器，返回 (img, ids)
#
#     Returns:
#         img_features_mu: [N, D] 嵌入均值
#         img_features_var: [N, D] 嵌入方差（可选）
#         ids: [N] 样本ID
#     """
#     model.eval()
#
#     # 使用 dummy input 推断输出维度
#     dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
#     with torch.no_grad():
#         dummy_output = model(img1=dummy_input,img1_type=img_type,state=state)
#
#     # 判断输出是否为 (mu, var) 元组
#     if isinstance(dummy_output, tuple) and len(dummy_output) == 2:
#         output_shape = dummy_output[0].shape[1:]  # mu 的特征维度
#         use_probabilistic = True
#     else:
#         output_shape = dummy_output.shape[1:]
#         use_probabilistic = False
#
#     N = len(dataloader.dataset)
#     batch_size = dataloader.batch_size
#
#     # 预分配内存
#     img_features_mu = torch.zeros((N, *output_shape), dtype=torch.float32, device=train_config.device)
#     ids = torch.zeros(N, dtype=torch.long, device=train_config.device)
#
#     if use_probabilistic:
#         img_features_var = torch.zeros((N, *output_shape), dtype=torch.float32, device=train_config.device)
#     else:
#         img_features_var = None
#
#     # 推理
#     with torch.no_grad(), autocast():
#         for i, (img, ids_current) in enumerate(tqdm(dataloader)):
#             img = img.to(train_config.device)
#             model_output = model(img1=img,img1_type=img_type,state=state)
#
#             if use_probabilistic:
#                 mu, var = model_output
#                 mu = F.normalize(mu, dim=-1)
#                 img_features_mu[i * batch_size : (i + 1) * batch_size] = mu
#                 img_features_var[i * batch_size : (i + 1) * batch_size] = var
#             else:
#                 mu = model_output
#                 if train_config.normalize_features:
#                     mu = F.normalize(mu, dim=-1)
#                 img_features_mu[i * batch_size : (i + 1) * batch_size] = mu
#
#             ids[i * batch_size : (i + 1) * batch_size] = ids_current
#
#     if use_probabilistic:
#         return (img_features_mu, img_features_var), ids
#     else:
#         return img_features_mu, ids


def predict(train_config, model, dataloader, img_type='query', state='mean'):
    """
    推理函数，支持模型输出为 (mu, var) 元组的概率嵌入模型

    Args:
        train_config: 包含 device, normalize_features 等配置
        model: 模型，输出为 (mu, var) 或 mu（兼容旧模型）
        dataloader: 数据加载器，返回 (img, ids)

    Returns:
        img_features_mu: [N, D] 嵌入均值
        img_features_var: [N, D] 嵌入方差（可选）
        ids: [N] 样本ID
    """
    model.eval()

    # 使用 dummy input 推断输出维度
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)

    with torch.no_grad():
        dummy_output = model(img1=dummy_input, state=state)

    # 判断输出是否为 (mu, var) 元组
    if isinstance(dummy_output, tuple) and len(dummy_output) == 2:
        output_shape = dummy_output[0].shape[1:]  # mu 的特征维度
        use_probabilistic = True
    else:
        output_shape = dummy_output.shape[1:]
        use_probabilistic = False

    N = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # 预分配内存
    img_features_mu = torch.zeros((N, *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(N, dtype=torch.long, device=train_config.device)

    if use_probabilistic:
        img_features_var = torch.zeros((N, *output_shape), dtype=torch.float32, device=train_config.device)
    else:
        img_features_var = None

    # 推理
    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            model_output = model(img1=img, state=state)

            if use_probabilistic:
                mu, var = model_output
                mu = F.normalize(mu, dim=-1)
                img_features_mu[i * batch_size: (i + 1) * batch_size] = mu
                img_features_var[i * batch_size: (i + 1) * batch_size] = var
            else:
                mu = model_output
                mu = F.normalize(mu, dim=-1)
                img_features_mu[i * batch_size: (i + 1) * batch_size] = mu

            ids[i * batch_size: (i + 1) * batch_size] = ids_current
    if use_probabilistic:
        return (img_features_mu, img_features_var), ids
    else:
        return img_features_mu, ids


# def predict_vigor(train_config, model, dataloader):
#     model.eval()

#     # Get output shape from a dummy input
#     dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
#     output_shape = model(dummy_input).shape[1:]

#     # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
#     total_samples = len(dataloader.dataset)
#     img_features = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
#     # Assuming each id_current has 4 elements, adjust the dimension for ids
#     ids = torch.zeros((total_samples, 4), dtype=torch.long, device=train_config.device)  # 修改为二维张量以匹配ids_current的形状

#     with torch.no_grad(), autocast():
#         for i, (img, ids_current) in enumerate(tqdm(dataloader)):
#             img = img.to(train_config.device)
#             img_feature = model(img)

#             # Normalize is calculated in fp32
#             if train_config.normalize_features:
#                 img_feature = F.normalize(img_feature, dim=-1)

#             img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
#             # Directly assign the 2D ids_current to the corresponding slice in ids
#             ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

#     return img_features, ids

# def predict(train_config, model, dataloader):
#     model.eval()
#
#     # wait before starting progress bar
#     time.sleep(0.1)
#
#     if train_config.verbose:
#         bar = tqdm(dataloader, total=len(dataloader))
#     else:
#         bar = dataloader
#
#     img_features_list = []
#
#     ids_list = []
#     with torch.no_grad():
#
#         for img, ids in bar:
#
#             ids_list.append(ids)
#
#             with autocast():
#
#                 img = img.to(train_config.device)
#                 img_feature = model(img)
#
#                 # normalize is calculated in fp32
#                 if train_config.normalize_features:
#                     img_feature = F.normalize(img_feature, dim=-1)
#
#             # save features in fp32 for sim calculation
#             img_features_list.append(img_feature.to(torch.float32))
#
#         # keep Features on GPU
#         img_features = torch.cat(img_features_list, dim=0)
#         ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
#
#     if train_config.verbose:
#         bar.close()
#
#     return img_features, ids_list