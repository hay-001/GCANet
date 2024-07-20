import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.main_utils import npy, v
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from functools import wraps
import time
from lapsolver import solve_dense
DIVISION_EPS = 1e-10
from softgroup.util import force_fp32
from softgroup.ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from collections import OrderedDict

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


def get_one_hot(targets, nb_classes):
    #res = np.eye(nb_classes)[np.array(targets, dtype=np.int8).reshape(-1)]
    # check none-type
    #if targets.min() == -1:
    #   idx = np.argwhere(targets == -1)
    #   res[idx] = 0
    #return res.reshape(list(targets.shape)+[nb_classes])

    one_hot = nn.functional.one_hot(targets, nb_classes)

    return one_hot


def hungarian_matching(W_pred, W_gt):
    # This non-tf function does not backprob gradient, only output matching indices
    # W_pred - NxK
    # W_gt - NxK'
    # Output: matching_indices
    # The matching does not include gt background instance
    # calculate RIoU
    n_points = W_pred.shape[0]
    #n_max_labels = min(W_gt.shape[1], W_pred.shape[1])
    #matching_indices = np.zeros([n_max_labels], dtype=np.int32)

    dot = np.sum(np.expand_dims(W_pred, axis=2) * np.expand_dims(W_gt, axis=1),
                 axis=0)  # K'xK
    denominator = np.expand_dims(np.sum(W_pred, axis=0),
                                 axis=1) + np.expand_dims(np.sum(W_gt, axis=0),
                                                          axis=0) - dot
    cost = dot / np.maximum(denominator, DIVISION_EPS)  # K'xK
    row_ind, col_ind = solve_dense(-cost)  # want max solution
    #matching_indices[b, :n_gt_labels] = col_ind

    return row_ind, col_ind

def compute_riou(W_pred, W_gt, pred_ind, gt_ind):
    # W_pred - NxK
    # W_gt - NxK'

    N, _ = W_pred.shape

    pred_ind = torch.LongTensor(pred_ind).unsqueeze(0).repeat(N, 1).to(
        W_pred.device)
    gt_ind = torch.LongTensor(gt_ind).unsqueeze(0).repeat(N, 1).to(W_gt.device)

    W_pred_reordered = torch.gather(W_pred, -1, pred_ind)
    W_gt_reordered = torch.gather(W_gt, -1, gt_ind)

    dot = torch.sum(W_gt_reordered * W_pred_reordered, dim=0)  # K
    denominator = torch.sum(W_gt_reordered, dim=0) + torch.sum(
        W_pred_reordered, dim=0) - dot
    mIoU = dot / (denominator + DIVISION_EPS)  # K
    return mIoU

def compute_miou(cluster_pred, I_gt):
    '''
    compute per-primitive riou loss
    cluster_pred: (1, N)
    I_gt: (1, N), must contains -1
    '''
    assert (cluster_pred.shape[0] == 1)

    one_hot_pred = get_one_hot(cluster_pred,
                               cluster_pred.max() + 1)[0]  # (N, K)

    if I_gt.min() == -1:
        one_hot_gt = get_one_hot(I_gt + 1,
                                 I_gt.max() +
                                 2)[0][:, 1:]  # (N, K'), remove background
    else:
        one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0]

    pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))

    riou = compute_riou(one_hot_pred, one_hot_gt, pred_ind, gt_ind)
    k = riou.shape[0]
    mean_riou = riou.sum() / k
    return mean_riou


def compute_type_miou(type_per_point, T_gt, cluster_pred, I_gt):
    '''
    compute per-primitive-instance type iou
    type_per_point: (1, N, K), K = 4
    T_gt: (1, N)
    '''
    assert (type_per_point.shape[0] == 1)
    
    # get T_pred: (1, N)
    if len(type_per_point.shape) == 3:
        B, N, _ = type_per_point.shape
        T_pred = torch.argmax(type_per_point, dim=-1) # (B, N)
    else:
        T_pred = type_per_point

   
    one_hot_pred = get_one_hot(cluster_pred,
                               cluster_pred.max() + 1)[0]  # (N, K)

    if I_gt.min() == -1:
        # (N, K'), remove background
        one_hot_gt = get_one_hot(I_gt + 1,
                                 I_gt.max() + 2)[0][:, 1:]  
    else:
        one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0]

    pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))
    type_iou = torch.Tensor([0.0]).to(T_gt.device)
    cnt = 0
    
    for p_ind, g_ind in zip(pred_ind, gt_ind):
        gt_type_label = T_gt[I_gt == g_ind].mode()[0]
        pred_type_label = T_pred[cluster_pred == p_ind].mode()[0]
        if gt_type_label == pred_type_label:
            type_iou += 1
        cnt += 1
    
    type_iou /= cnt
    return type_iou

def compute_type_miou_abc(type_per_point, T_gt, cluster_pred, I_gt):
    '''
    compute per-primitive-instance type iou
    type_per_point: (1, N, K), K = 6/10
    T_gt: (1, N)
    '''
    assert (type_per_point.shape[0] == 1)
    
    # get T_pred: (1, N)
    if len(type_per_point.shape) == 3:
        B, N, _ = type_per_point.shape
        T_pred = torch.argmax(type_per_point, dim=-1) # (B, N)
    else:
        T_pred = type_per_point
     
    T_pred[T_pred == 6] = 0
    T_pred[T_pred == 7] = 0
    T_pred[T_pred == 9] = 0
    T_pred[T_pred == 8] = 2
    
    T_gt[T_gt == 6] = 0
    T_gt[T_gt == 7] = 0
    T_gt[T_gt == 9] = 0
    T_gt[T_gt == 8] = 2
   
    one_hot_pred = get_one_hot(cluster_pred,
                               cluster_pred.max() + 1)[0]  # (N, K)

    if I_gt.min() == -1:
        # (N, K'), remove background
        one_hot_gt = get_one_hot(I_gt + 1,
                                 I_gt.max() + 2)[0][:, 1:]  
    else:
        one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0]

    pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))
    type_iou = torch.Tensor([0.0]).to(T_gt.device)
    cnt = 0
    riou = compute_riou(one_hot_pred, one_hot_gt, pred_ind, gt_ind)
    for p_ind, g_ind in zip(pred_ind, gt_ind):
        gt_type_label = T_gt[I_gt == g_ind].mode()[0]
        try:
            pred_type_label = T_pred[cluster_pred == p_ind].mode()[0]
        except:
            continue
        if gt_type_label == pred_type_label:
            type_iou += 1
           
        cnt += 1
    
    type_iou /= cnt
    return type_iou

def compute_embedding_loss(pred_feat, gt_label, t_pull=0.5, t_push=1.5):
    '''
    pred_feat: (B, N, K)
    gt_label: (B, N)
    '''
    batch_size, num_pts, feat_dim = pred_feat.shape
    device = pred_feat.device
    pull_loss = torch.Tensor([0.0]).to(device)
    push_loss = torch.Tensor([0.0]).to(device)
    for i in range(batch_size):
        num_class = gt_label[i].max() + 2

        embeddings = []

        for j in range(num_class):
            mask = (gt_label[i] == (j - 1))
            feature = pred_feat[i][mask]
            if len(feature) == 0:
                continue
            embeddings.append(feature)  # (M, K)

        centers = []

        for feature in embeddings:
            center = torch.mean(feature, dim=0).view(1, -1)
            centers.append(center)

        # intra-embedding loss
        pull_loss_tp = torch.Tensor([0.0]).to(device)
        for feature, center in zip(embeddings, centers):
            dis = torch.norm(feature - center, 2, dim=1) - t_pull
            dis = F.relu(dis)
            pull_loss_tp += torch.mean(dis)

        pull_loss = pull_loss + pull_loss_tp / len(embeddings)

        # inter-embedding loss
        centers = torch.cat(centers, dim=0)  # (num_class, K)

        if centers.shape[0] == 1:
            continue

        dst = torch.norm(centers[:, None, :] - centers[None, :, :], 2, dim=2)

        eye = torch.eye(centers.shape[0]).to(device)
        pair_distance = torch.masked_select(dst, eye == 0)

        pair_distance = t_push - pair_distance
        pair_distance = F.relu(pair_distance)
        push_loss += torch.mean(pair_distance)

    pull_loss = pull_loss / batch_size
    push_loss = push_loss / batch_size
    loss = pull_loss + push_loss
    return loss, pull_loss, push_loss

def k_means(p, num_class=2):
    b, N, c = p.shape

    IDX = torch.zeros(b, N).to(p.device).long()

    for i in range(b):
        cur_feat = npy(p[i])
        kmeans = KMeans(n_clusters=num_class, random_state=0).fit(cur_feat)
        IDX[i] = v(kmeans.labels_)

    return IDX

def compute_normal_loss(pred, gt):

    b, N, _ = pred.shape
    normal_loss = torch.acos((pred * gt).sum(-1).clamp(-0.99, 0.99))

    normal_loss = normal_loss.sum() / (b * N)

    return normal_loss

def compute_type_loss(pred, gt):
    '''
    pred: (B, N, K)
    gt: (B, N)
    '''

    type_loss = nn.CrossEntropyLoss()
    valid_class = (gt != -1)  # remove background
    gt = gt[valid_class]

    pred = pred[valid_class]

    loss = type_loss(pred, gt)
    
    return loss


def offset_loss(pt_offsets, instance_labels,pt_offset_labels):
    ignore_label=-1
    pos_inds = instance_labels != ignore_label
    pos_inds=pos_inds.squeeze(1)
    if pos_inds.sum() == 0:
        offset_loss = 0 * pt_offsets.sum()
    else:
        offset_loss = F.l1_loss(
            pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
    return offset_loss

def instance_loss(cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                  instance_labels, instance_pointnum, instance_cls, instance_batch_idxs,instance_classes=10):
    ignore_label=0
    instance_classes=instance_classes-1#9
    pos_iou_thr=0.5
    losses = {}


    if proposals_idx.size(0) == 0 or (instance_cls != ignore_label).sum() == 0:
        cls_loss = cls_scores.sum() * 0
        mask_loss = mask_scores.sum() * 0
        iou_score_loss = iou_scores.sum() * 0

        losses['cls_loss'] = cls_loss
        losses['mask_loss'] = mask_loss
        losses['iou_score_loss'] = iou_score_loss
        losses['num_pos'] = mask_loss
        losses['num_neg'] = mask_loss

        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        return loss

    proposals_idx = proposals_idx[:, 1].int().cuda()
    proposals_offset = proposals_offset.cuda()

    # cal iou of clustered instance
    ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                              instance_pointnum)



    # filter out background instances
    fg_inds = (instance_cls != ignore_label)
    fg_instance_cls = instance_cls[fg_inds]
    fg_ious_on_cluster = ious_on_cluster[:, fg_inds]



    # assign proposal to gt idx. -1: negative, 0 -> num_gts - 1: positive
    num_proposals = fg_ious_on_cluster.size(0)
    num_gts = fg_ious_on_cluster.size(1)
    assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals, ), -1, dtype=torch.long)
    # overlap > thr on fg instances are positive samples
    max_iou, argmax_iou = fg_ious_on_cluster.max(1)
    pos_inds = max_iou >= pos_iou_thr
    assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]





    # allow low-quality proposals with best iou to be as positive sample
    # in case pos_iou_thr is too high to achieve
    match_low_quality = False
    min_pos_thr = 0
    if match_low_quality:
        gt_max_iou, gt_argmax_iou = fg_ious_on_cluster.max(0)
        for i in range(num_gts):
            if gt_max_iou[i] >= min_pos_thr:
                assigned_gt_inds[gt_argmax_iou[i]] = i
    # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
    labels = fg_instance_cls.new_full((num_proposals, ), instance_classes)

    pos_inds = assigned_gt_inds >= 0
    labels[pos_inds] = fg_instance_cls[assigned_gt_inds[pos_inds]]
    cls_loss = F.cross_entropy(cls_scores, labels)
    losses['cls_loss'] = cls_loss



    # compute mask loss
    mask_cls_label = labels[instance_batch_idxs.long()]
    slice_inds = torch.arange(
        0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
    mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]



    mask_label = get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                                instance_pointnum, ious_on_cluster, pos_iou_thr)

    mask_label_weight = (mask_label != -1).float()
    mask_label[mask_label == -1.] = 0.5  # any value is ok
    mask_loss = F.binary_cross_entropy(
        mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
    mask_loss /= (mask_label_weight.sum() + 1)
    losses['mask_loss'] = mask_loss



    # compute iou score loss
    ious = get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                instance_pointnum, mask_scores_sigmoid_slice.detach())
    fg_ious = ious[:, fg_inds]
    gt_ious, _ = fg_ious.max(1)
    slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
    iou_score_weight = (labels < instance_classes).float()
    iou_score_slice = iou_scores[slice_inds, labels]
    iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')
    iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
    losses['iou_score_loss'] = iou_score_loss

    # add logging variables
    losses['num_pos'] = (labels < instance_classes).sum().float()
    losses['num_neg'] = (labels >= instance_classes).sum().float()

    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    return loss





def compute_nnl_loss(pred, gt):
    '''
    pred: (B, N, K)
    gt: (B, N)
    '''

    type_loss = nn.NLLLoss()
    valid_class = (gt != -1)  # remove background
    gt = gt[valid_class]

    pred = pred[valid_class]

    loss = type_loss(pred, gt)
    
    return loss


def compute_instance_loss(pred, gt):
    '''
    pred: (B, N, K)
    gt: (B, N)
    '''
    type_loss = nn.CrossEntropyLoss()
    valid_class = (gt != -1)  # remove background

    gt = gt[valid_class]

    pred = pred[valid_class]

    loss = type_loss(pred, gt)
    
    return loss

def compute_param_loss(pred, T_gt, T_param_gt):
    '''
    only add loss to corresponding type
    pred: (B, N, 22)
    T_gt: (B, N)
    T_param_gt: (B, N, 22)
    '''
    param_list = {5:[0,4], 1:[4,8], 4:[8,15], 3:[15,22]}

    #[0, 4, 8, 15, 22]

    b, N, _ = pred.shape
    
    #l2_loss = nn.MSELoss(reduction='sum')
    l2_loss = nn.MSELoss()

    total_loss = 0
    length = 0
    cnt = 0
    for b in range(pred.shape[0]):
        for i in [1, 4, 5, 3]:
            index = T_gt[b] == i
            tmp_pred = pred[b][index]
            tmp_gt = T_param_gt[b][index]

            if tmp_pred.shape[0] == 0:
                continue
            if tmp_gt.sum() == 0: # no parameters to process
                continue

            tmp_pred = tmp_pred[:, param_list[i][0]:param_list[i][1]]
            tmp_gt = tmp_gt[:, param_list[i][0]:param_list[i][1]].float()
            
            valid_mask = tmp_gt.sum(1) != 0

            tmp_pred = tmp_pred[valid_mask]
            tmp_gt = tmp_gt[valid_mask]
            
            if tmp_gt.shape[0] == 0:
                continue

            tmp_loss = l2_loss(tmp_pred, tmp_gt)

            # ignore wrong type label 
            if tmp_gt.max() > 10 or tmp_loss > 50: 
                continue

            total_loss += tmp_loss
            
            length += tmp_pred.shape[0]
            cnt += 1

    #TODO: only happened in test phase
    if cnt == 0:
        if torch.isnan(l2_loss(tmp_pred, tmp_gt.float())).sum() > 0:
            return torch.Tensor([0.0]).to(T_gt.device)
            # return torch.Tensor(0.0).to(T_gt.device)
        return l2_loss(tmp_pred, tmp_gt.float())

    total_loss = total_loss / cnt

    return total_loss
