import os
import sys
import numpy as np
import torch
import torch.nn as nn
# import pointgroup_ops
import spconv.pytorch as spconv
from .backbone import ResidualBlock, UBlock, MLP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import functools
from softgroup.util import force_fp32, rle_decode, rle_encode
from softgroup.ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                           get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                           voxelization_idx, hierarchical_aggregation)

from models.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import grouping_operation

from models.transformer import Transformer

from models.search_knn import knn_point, group_points
from utils.abc_utils import construction_affinity_matrix_type, \
    construction_affinity_matrix_normal, construction_affinity_matrix_type_one_class


def knn(x, k1, k2):
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx


def knn_points_normals(x, k1, k2):
    """
    The idea is to design the distance metric for computing
    nearest neighbors such that the normals are not given
    too much importance while computing the distances.
    Note that this is only used in the first layer.
    """
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            p = x[b: b + 1, 0:3]
            n = x[b: b + 1, 3:6]

            inner = 2 * torch.matmul(p.transpose(2, 1), p)
            xx = torch.sum(p ** 2, dim=1, keepdim=True)
            p_pairwise_distance = xx - inner + xx.transpose(2, 1)

            inner = 2 * torch.matmul(n.transpose(2, 1), n)
            n_pairwise_distance = 2 - inner

            # This pays less attention to normals
            pairwise_distance = p_pairwise_distance * (1 + n_pairwise_distance)

            # This pays more attention to normals
            # pairwise_distance = p_pairwise_distance * torch.exp(n_pairwise_distance)

            # pays too much attention to normals
            # pairwise_distance = p_pairwise_distance + n_pairwise_distance

            distances.append(-pairwise_distance)

        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx


def get_graph_feature(x, k1=20, k2=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k1=k1, k2=k2)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


def get_graph_feature_with_normals(x, k1=20, k2=20, idx=None):
    """
    normals are treated separtely for computing the nearest neighbor
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn_points_normals(x, k1=k1, k2=k2)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


def get_graph_feature_with_normals_g(x, k1=20, k2=20, idx=None):
    """
    normals are treated separtely for computing the nearest neighbor
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn_points_normals(x, k1=k1, k2=k2)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)


    # n_sub = torch.gather(normal, -1, nnid.view(B, 1, -1).repeat(1, 3, 1)).view(B, 3, -1, k)   b 3 n 1


    angle = (x.permute(0, 2, 1).unsqueeze(-1)[:,3:6,:,:] * feature.permute(0,3, 1, 2)[:,3:6,:,:]).sum(1).clamp(-0.99, 0.99)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((angle.unsqueeze(-1),feature[:,:,:,3:6] - x[:,:,:,3:6], x[:,:,:,3:6]), dim=3).permute(0, 3, 1, 2)
    return feature




def compute_batch_adjacency_matrix(batch_point_clouds, radius=0, dist_state=True, sigma=1.0):
    # 计算点云中每个点之间的欧几里德距离
    distances = torch.cdist(batch_point_clouds, batch_point_clouds)
    # 将距离矩阵转换为相邻矩阵
    if dist_state:
        adjacency = distances
    else:
        adjacency = (distances <= radius).float()
    # 将对角线上的元素设置为0
    adjacency = adjacency - torch.diag_embed(torch.diagonal(adjacency, dim1=-2, dim2=-1))
    # # 将每行元素归一化到[0,1]范围内
    # min_dist = adjacency.min(dim=-1, keepdim=True)[0]
    # max_dist = adjacency.max(dim=-1, keepdim=True)[0]
    # adjacency = (adjacency - min_dist) / (max_dist - min_dist)
    # 将距离值整体归一化到[0,1]范围内
    min_dist = adjacency.min()
    max_dist = adjacency.max()
    adjacency = (adjacency - min_dist) / (max_dist - min_dist)

    # 将距离值转换为相似性概率
    adjacency = torch.exp(-adjacency ** 2 / (2 * sigma ** 2))
    adjacency = adjacency - torch.diag_embed(torch.diagonal(adjacency, dim1=-2, dim2=-1))

    return adjacency


def neighborhood_pointwise_attention(point_cloud, neighborhood, be_weighted_point_cloud):
    """
    计算点云中每个点与其邻域点之间的距离，然后将距离进行softmax激活后，使用pointwise的方式计算邻域点特征的attention加权和。

    Args:
        point_cloud (torch.Tensor): 点云数据, shape [B, N, C]
        neighborhood (torch.Tensor): 每个点的K个邻域点的数据, shape [B, N, K, C]
        p (float, optional): 使用的距离的p值. Defaults to 2.0.

    Returns:
        torch.Tensor: 点云中每个点的attention加权特征, shape [B, N, C]
    """
    B, N, K, C = neighborhood.shape

    # 计算每个点与邻域点之间的距离

    distances = torch.cdist(point_cloud.unsqueeze(2), neighborhood, p=2)
    # distances: [B, N, K, K]
    # print(distances.shape)

    # 对距离进行softmax操作
    weights = F.softmax(-distances, dim=-1)
    # weights: [B, N, K, K]

    # 计算每个点的邻域特征的加权和

    # weights=weights.permute(0,1,3,2).repeat(1,1,1,be_weighted_point_cloud.shape[3])
    # print(be_weighted_point_cloud.shape)
    # print('weights.shape:',weights.shape)
    # # neighborhood_features: weights: [B, N, K, C]
    # neighborhood_features = torch.mul(weights,be_weighted_point_cloud)
    # print('neighborhood_features:',neighborhood_features.shape)
    #
    # return neighborhood_features

    weights = weights.permute(0, 1, 3, 2).repeat(1, 1, 1, be_weighted_point_cloud.shape[3])

    return weights


def inst_and_seg_attention(semantic_feature, semantic_feature_knn, instance_feature, instance_feature_knn, feature_knn):
    """
    计算点云中每个点与其邻域点之间的距离，然后将距离进行softmax激活后，使用pointwise的方式计算邻域点特征的attention加权和。

    Args:
        point_cloud (torch.Tensor): 点云数据, shape [B, N, C]
        neighborhood (torch.Tensor): 每个点的K个邻域点的数据, shape [B, N, K, C]
        p (float, optional): 使用的距离的p值. Defaults to 2.0.

    Returns:
        torch.Tensor: 点云中每个点的attention加权特征, shape [B, N, C]
    """
    # 计算每个点与邻域点之间的距离

    distances_semantic = torch.cdist(semantic_feature.unsqueeze(2), semantic_feature_knn, p=2)
    distances_instance = torch.cdist(instance_feature.unsqueeze(2), instance_feature_knn, p=2)

    # 对距离进行softmax操作
    weights_semantic = F.softmax(-distances_semantic, dim=-1)
    weights_instance = F.softmax(-distances_instance, dim=-1)

    weights = weights_semantic + weights_instance
    weights = F.softmax(weights, dim=-1)

    weights = weights.permute(0, 1, 3, 2).repeat(1, 1, 1, feature_knn.shape[3])

    feature_knn_s_i = torch.mul(weights, feature_knn).permute(0, 3, 2, 1)

    return feature_knn_s_i


def inst_and_seg_dist(semantic_feature, semantic_feature_knn, instance_feature, instance_feature_knn):
    """
    计算点云中每个点与其邻域点之间的距离，然后将距离进行softmax激活后，使用pointwise的方式计算邻域点特征的attention加权和。

    Args:
        point_cloud (torch.Tensor): 点云数据, shape [B, N, C]
        neighborhood (torch.Tensor): 每个点的K个邻域点的数据, shape [B, N, K, C]
        p (float, optional): 使用的距离的p值. Defaults to 2.0.

    Returns:
        torch.Tensor: 点云中每个点的attention加权特征, shape [B, N, C]
    """
    # 计算每个点与邻域点之间的距离
    distances_semantic = torch.cdist(semantic_feature.unsqueeze(2), semantic_feature_knn, p=2)
    distances_instance = torch.cdist(instance_feature.unsqueeze(2), instance_feature_knn, p=2)

    return distances_semantic, distances_instance


def cos_dist(instance_feature, global_instance_feature):
    """
    计算点云中每个点与全局实例特征中每个点之间的余弦相似度距离。

    Args:
        instance_feature (torch.Tensor): 实例特征, shape [B, N, C]
        global_instance_feature (torch.Tensor): 全局实例特征, shape [B, K, C]

    Returns:
        torch.Tensor: 实例特征余弦相似度距离, shape [B, N, K]
    """
    # 计算实例特征余弦相似度距离
    instance_feature_norm = instance_feature.norm(dim=-1, keepdim=True)
    global_instance_feature_norm = global_instance_feature.norm(dim=-1, keepdim=True)
    instance_cos_dist = 1 - torch.einsum('bnc,bkc->bnk', instance_feature / instance_feature_norm, global_instance_feature / global_instance_feature_norm)
    instance_cos_dist=-instance_cos_dist
    return instance_cos_dist








class KPAM(nn.Module):
    def __init__(self, C):
        super(KPAM, self).__init__()
        self.dim = C

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=False)
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, attention_feature):
        attention_feature = attention_feature.permute(0, 2, 1)
        attention_feature = self.conv1(attention_feature).permute(0, 2, 1)  # b,c,n
        attention_feature = self.softmax(attention_feature).unsqueeze(-1)  # b,n,c

        attention_matrix = attention_feature.repeat(1, 1, 1, x.shape[-1])

        out = torch.mul(attention_matrix, x)

        return out


class OFFSET_PRED_MODULE(nn.Module):
    def __init__(self, nn_nb=30, sampling_ratio=120):
        super(OFFSET_PRED_MODULE, self).__init__()
        self.k = nn_nb
        self.dilation_factor = 1
        self.drop = 0.0
        self.sampling_ratio = sampling_ratio

        # self.transformer = Transformer(dim=128, depth=1,
        #                               heads = 4, dim_head = 64,
        #                            mlp_dim=64, dropout=0)

        self.bn1 = nn.GroupNorm(2, 128)

        self.conv1 = nn.Sequential(nn.Conv2d(131, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.attention = KPAM(nn_nb)

        self.mlp_offset = torch.nn.Conv1d(256, 3, 1)

    def forward(self, points, feature, instance_feature):


        batch_size, num_points, _ = points.shape

        l = np.arange(num_points)
        np.random.seed(1234)
        np.random.shuffle(l)
        random_index = torch.from_numpy(l[:self.sampling_ratio])

        subidx = random_index.to(points.device).long().view(1, -1, 1).repeat(batch_size, 1, 1)  # 点的序号

        key_points = torch.gather(points, 1, subidx.repeat(1, 1, points.shape[2]))  # [b, n_sub, c]

        feature_sampling = torch.gather(feature, 1, subidx.repeat(1, 1, feature.shape[2]))  # [b, n_sub, c]
        instance_feature_sampling = torch.gather(instance_feature, 1, subidx.repeat(1, 1, instance_feature.shape[2]))  # [b, n_sub, c]


        distances_instance=cos_dist(instance_feature, instance_feature_sampling)# [b, n, n_sub]




        topk_dist = torch.topk(distances_instance,self.k, dim=2, largest=True)[0]  # 获取最小距离的索引[b, n, k]
        topk_idx = torch.topk(distances_instance,self.k, dim=2, largest=True)[1]  # 获取最小距离的索引[b, n, k]


        topk_key_points = torch.gather(key_points.unsqueeze(1).repeat(1, topk_idx.shape[1], 1, 1), 
                                       dim=2, index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, key_points.shape[2]))#[b, n, k,3]
        

        topk_feature_sampling = torch.gather(feature_sampling.unsqueeze(1).repeat(1, topk_idx.shape[1], 1, 1), 
                                             dim=2, index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, feature_sampling.shape[2]))#[b, n, k,64]
        




        points_knn_direction=topk_key_points-points.unsqueeze(2).expand(-1, -1,self.k, -1)
        feature_knn_plus_direction = torch.cat([topk_feature_sampling, points_knn_direction], 3)#[b, n, k,128+3]
 


        feature_attention = self.attention(feature_knn_plus_direction, topk_dist)



        feature_attention = self.conv1(feature_attention.permute(0, 3, 2, 1))
        feature_attention = feature_attention.max(dim=-2, keepdim=False)[0]
        feature_attention = torch.cat([feature_attention, feature.permute(0, 2, 1)], dim=1)

        offsets = self.mlp_offset(feature_attention)
        # print('offsets',offsets.shape)

        return offsets


class DGCNNEncoderGn(nn.Module):
    def __init__(self, mode=0, nn_nb=80, input_channels=3,
                 ):
        super(DGCNNEncoderGn, self).__init__()
        self.k = nn_nb
        self.dilation_factor = 1
        self.mode = mode
        self.drop = 0.0
        self.bn1 = nn.GroupNorm(2, 64)
        self.bn2 = nn.GroupNorm(2, 64)
        self.bn3 = nn.GroupNorm(2, 128)
        self.bn4 = nn.GroupNorm(4, 256)
        self.bn5 = nn.GroupNorm(8, 1024)
        if mode==5:
            self.conv1 = nn.Sequential(nn.Conv2d(input_channels * 2, 64, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=1, bias=False),
                            self.bn1,
                            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                    self.bn2,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                    self.bn3,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.mlp1 = nn.Conv1d(256, 1024, 1)
        self.bnmlp1 = nn.GroupNorm(8, 1024)
        self.mlp1 = nn.Conv1d(256, 1024, 1)
        self.bnmlp1 = nn.GroupNorm(8, 1024)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.shape[2]

        if self.mode == 5:
            x = get_graph_feature_with_normals(x, k1=self.k, k2=self.k)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x_features = torch.cat((x1, x2, x3), dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))
            x4 = x.max(dim=2)[0]
            x4 = x4.view(batch_size, 1024, 1).repeat(1, 1, num_points)
            output_feats = torch.cat([x4, x_features], 1)
        else:
            # First edge conv
            x = get_graph_feature(x, k1=self.k, k2=self.k)

            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x_features = torch.cat((x1, x2, x3), dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))
            x4 = x.max(dim=2)[0]
            x4 = x4.view(batch_size, 1024, 1).repeat(1, 1, num_points)
            output_feats = torch.cat([x4, x_features], 1)
        return output_feats


class PrimitivesEmbeddingDGCNGn(nn.Module):
    """
    Segmentation model that takes point cloud as input and returns per
    point embedding or membership function. This defines the membership loss
    inside the forward function so that data distributed loss can be made faster.
    """

    def __init__(self, opt, emb_size=64, num_primitives=10, primitives=True, embedding=True, parameters=True, mode=5,
                 num_channels=3, nn_nb=80):
        super(PrimitivesEmbeddingDGCNGn, self).__init__()
        self.opt = opt
        self.mode = mode
        self.nn_nb = nn_nb
        self.encoder = DGCNNEncoderGn(mode=mode, nn_nb=nn_nb,input_channels=num_channels)

        self.offset_pred_block = OFFSET_PRED_MODULE(nn_nb=30, sampling_ratio=120)  # 30,120

        self.drop = 0.0

        self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        self.bn1 = nn.GroupNorm(8, 512)

        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.bn2 = nn.GroupNorm(4, 256)
        if mode==5 or mode==3:
            self.conv3 = torch.nn.Conv1d(262, 128, 1)
        else:
            self.conv3 = torch.nn.Conv1d(259, 128, 1)

        self.bn3 = nn.GroupNorm(4, 128)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding
        self.parameters = parameters
        self.max_proposal_num = 200
        self.semantic_classes = num_primitives


        if self.embedding:
            self.mlp_seg_prob1 = torch.nn.Conv1d(832, 256, 1)
            self.mlp_seg_prob2 = torch.nn.Conv1d(256, self.emb_size, 1)
            self.bn_seg_prob1 = nn.GroupNorm(4, 256)

            self.bn_normal = nn.GroupNorm(2, 64)
            self.conv_normal = nn.Sequential(nn.Conv2d(7, 64, kernel_size=1, bias=False),
                                        self.bn_normal,
                                        nn.LeakyReLU(negative_slope=0.2))

        # 逐点分割
        if primitives:
            self.mlp_prim_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_prim_prob2 = torch.nn.Conv1d(256, num_primitives, 1)
            self.bn_prim_prob1 = nn.GroupNorm(4, 256)
        # 逐点基元参数
        if parameters:
            self.mlp_param_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_param_prob2 = torch.nn.Conv1d(256, 22, 1)
            self.bn_param_prob1 = nn.GroupNorm(4, 256)
        # 逐点法向
        if self.mode == 3:
            self.mlp_normal_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_normal_prob2 = torch.nn.Conv1d(256, 3, 1)
            self.bn_normal_prob1 = nn.GroupNorm(4, 256)

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.channels = 64
        if self.opt.ablation:
            self.offset_linear = MLP(128, 3, norm_fn=norm_fn, num_layers=2)

        self.tiny_unet = UBlock([self.channels, 2 * self.channels], norm_fn, 2, block=ResidualBlock, indice_key_id=11)
        self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(self.channels), nn.ReLU())
        self.cls_linear = nn.Linear(self.channels, self.semantic_classes)
        self.mask_linear = MLP(self.channels, self.semantic_classes, norm_fn=None, num_layers=2)
        self.iou_score_linear = nn.Linear(self.channels, self.semantic_classes)


        # if mode==5:
        #     self.mlp_squeeze_output_feature = torch.nn.Conv1d(102, self.channels, 1)
        # else:
        #     self.mlp_squeeze_output_feature = torch.nn.Conv1d(99, self.channels, 1)
        if num_primitives!=7:
            self.mlp_squeeze_output_feature = torch.nn.Conv1d(102, self.channels, 1)
        else:
            self.mlp_squeeze_output_feature = torch.nn.Conv1d(99, self.channels, 1)
        self.bn_normal_squeeze_output_feature = nn.GroupNorm(4, self.channels)

    def forward(self, points, normals, v2p_maps, batch_idxs, postprocess=False):
        if postprocess:
            return self.forward_test(points, normals,v2p_maps, batch_idxs)
        else:
            return self.forward_train(points, normals, batch_idxs)

    def forward_train(self, points, normals, batch_idxs):

        batch_size, N, _ = points.shape

        if self.mode == 5:
            points = torch.cat([points, normals], dim=-1)

        points = points.permute(0, 2, 1)
        x = self.encoder(points)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)


        # 逐点分割
        if self.primitives:
            x_type = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
            type_per_point = self.mlp_prim_prob2(x_type)

            if 'r' in self.opt.loss_class:
                type_per_point_forgroup = type_per_point.permute(0, 2, 1)
                type_per_point = self.logsoftmax(type_per_point).permute(0, 2, 1)
            else:
                type_per_point = type_per_point.permute(0, 2, 1)

        # 逐点基元参数
        if self.parameters:
            x = F.dropout(F.relu(self.bn_param_prob1(self.mlp_param_prob1(x_all))), self.drop)
            x_para=x
            param_per_point = self.mlp_param_prob2(x).transpose(1, 2)
            sphere_param = param_per_point[:, :, :4]
            plane_norm = torch.norm(param_per_point[:, :, 4:7], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
            plane_normal = param_per_point[:, :, 4:7] / plane_norm
            plane_param = torch.cat([plane_normal, param_per_point[:, :, 7:8]], dim=2)
            cylinder_norm = torch.norm(param_per_point[:, :, 8:11], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
            cylinder_normal = param_per_point[:, :, 8:11] / cylinder_norm
            cylinder_param = torch.cat([cylinder_normal, param_per_point[:, :, 11:15]], dim=2)

            cone_norm = torch.norm(param_per_point[:, :, 15:18], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
            cone_normal = param_per_point[:, :, 15:18] / cone_norm
            cone_param = torch.cat([cone_normal, param_per_point[:, :, 18:22]], dim=2)

            param_per_point = torch.cat([sphere_param, plane_param, cylinder_param, cone_param], dim=2)




        # 逐点法向
        if self.mode == 3:
            x = F.dropout(F.relu(self.bn_normal_prob1(self.mlp_normal_prob1(x_all))), self.drop)
            normal_per_point = self.mlp_normal_prob2(x).permute(0, 2, 1)
            normal_norm = torch.norm(normal_per_point, dim=-1,
                                 keepdim=True).repeat(1, 1, 3) + 1e-12
            normal_per_point = normal_per_point / normal_norm
            points = torch.cat([points, normal_per_point.permute(0, 2, 1)], dim=1)

        if self.embedding:
            normal_feature = get_graph_feature_with_normals_g(points, k1=self.nn_nb, k2=self.nn_nb)
            normal_feature = self.conv_normal(normal_feature)
            normal_feature = normal_feature.max(dim=-1, keepdim=False)[0]


            
            x = torch.cat([x_all, x_type, x_para, normal_feature], dim=1)
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x))), self.drop)
            output_feats = self.mlp_seg_prob2(x).permute(0, 2, 1)  # 逐点特征


        coords_float = points.permute(0, 2, 1)


        # coords_float = torch.reshape(coords_float, (-1, coords_float.shape[-1]))
        coords_float = coords_float.reshape(-1, coords_float.shape[-1])

        if self.mode >0:
            coords_float = coords_float[:, 0:3]
        output_feats_puls_coods = torch.cat([x_all.permute(0, 2, 1), points.permute(0, 2, 1)], dim=-1)
        # pt_offsets = self.offset_linear(output_feats_puls_coods)

        output_feats_puls_coods = F.dropout(F.relu(self.bn3(self.conv3(output_feats_puls_coods.permute(0, 2, 1)))),
                                            self.drop)

        output_feats_puls_coods = output_feats_puls_coods.permute(0, 2, 1)
        # semantic_scores=type_per_point.permute(0,2,1)

        # semantic_scores = torch.reshape(semantic_scores, (batch_size*N, -1))

        semantic_scores = torch.reshape(type_per_point_forgroup, (-1, type_per_point_forgroup.shape[-1]))

        if self.opt.ablation!=True:
            pt_offsets = self.offset_pred_block(points[:, 0:3, :].permute(0, 2, 1), output_feats_puls_coods
                                                , output_feats)
            pt_offsets = pt_offsets.permute(0, 2, 1)
            pt_offsets = torch.reshape(pt_offsets, (-1, pt_offsets.shape[-1]))
            # pt_offsets = pt_offsets.view(-1, pt_offsets.shape[-1])

        else:
            output_feats_puls_coods = torch.reshape(output_feats_puls_coods, (-1, output_feats_puls_coods.shape[-1]))
            pt_offsets = self.offset_linear(output_feats_puls_coods)


        # proposals_idx, proposals_offset = \
        #     self.forward_grouping(semantic_scores, pt_offsets,batch_idxs, coords_float)
        proposals_idx, proposals_offset = self.forward_grouping(
            semantic_scores,
            pt_offsets,
            batch_idxs,
            coords_float,
            type_per_point,
            param_per_point,
            output_feats,
            lvl_fusion=False,
            training_mode='train'
            )

        if proposals_offset.shape[0] > self.max_proposal_num:
            proposals_offset = proposals_offset[:self.max_proposal_num + 1]
            proposals_idx = proposals_idx[:proposals_offset[-1]]
            assert proposals_idx.shape[0] == proposals_offset[-1]




        output_feats_reshape = torch.reshape(output_feats, (-1, output_feats.shape[-1]))
        # output_feats_reshape = output_feats.view(-1, output_feats.shape[-1])




      
        inst_feats, inst_map = self.clusters_voxelization(
            proposals_idx,
            proposals_offset,
            output_feats_reshape,
            coords_float,
            rand_quantize=True,
            spatial_shape=64,
            scale=64,
        )
        instance_batch_idxs, cls_scores, iou_scores, mask_scores = self.forward_instance(
            inst_feats, inst_map)
        if self.mode==3:
            return type_per_point, param_per_point,normal_per_point, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
            , proposals_idx, proposals_offset, output_feats
        else:
            return type_per_point, param_per_point, \
                    semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                , proposals_idx, proposals_offset, output_feats


    def forward_test(self, points, normals,v2p_maps, batch_idxs):

        batch_size, N, _ = points.shape

        if self.mode == 5:
            points = torch.cat([points, normals], dim=-1)

        points = points.permute(0, 2, 1)
        x = self.encoder(points)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)


        # 逐点分割
        if self.primitives:
            x_type = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
            type_per_point = self.mlp_prim_prob2(x_type)

            if 'r' in self.opt.loss_class:
                type_per_point_forgroup = type_per_point.permute(0, 2, 1)
                type_per_point = self.logsoftmax(type_per_point).permute(0, 2, 1)
            else:
                type_per_point = type_per_point.permute(0, 2, 1)

        # 逐点基元参数
        if self.parameters:
            x = F.dropout(F.relu(self.bn_param_prob1(self.mlp_param_prob1(x_all))), self.drop)
            x_para=x
            param_per_point = self.mlp_param_prob2(x).transpose(1, 2)
            sphere_param = param_per_point[:, :, :4]
            plane_norm = torch.norm(param_per_point[:, :, 4:7], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
            plane_normal = param_per_point[:, :, 4:7] / plane_norm
            plane_param = torch.cat([plane_normal, param_per_point[:, :, 7:8]], dim=2)
            cylinder_norm = torch.norm(param_per_point[:, :, 8:11], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
            cylinder_normal = param_per_point[:, :, 8:11] / cylinder_norm
            cylinder_param = torch.cat([cylinder_normal, param_per_point[:, :, 11:15]], dim=2)

            cone_norm = torch.norm(param_per_point[:, :, 15:18], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
            cone_normal = param_per_point[:, :, 15:18] / cone_norm
            cone_param = torch.cat([cone_normal, param_per_point[:, :, 18:22]], dim=2)

            param_per_point = torch.cat([sphere_param, plane_param, cylinder_param, cone_param], dim=2)



        # 逐点法向
        if self.mode == 3:
            x = F.dropout(F.relu(self.bn_normal_prob1(self.mlp_normal_prob1(x_all))), self.drop)
            normal_per_point = self.mlp_normal_prob2(x).permute(0, 2, 1)
            normal_norm = torch.norm(normal_per_point, dim=-1,
                                 keepdim=True).repeat(1, 1, 3) + 1e-12
            normal_per_point = normal_per_point / normal_norm
            points = torch.cat([points, normal_per_point.permute(0, 2, 1)], dim=1)




        if self.embedding:
            normal_feature = get_graph_feature_with_normals_g(points, k1=self.nn_nb, k2=self.nn_nb)
            normal_feature = self.conv_normal(normal_feature)
            normal_feature = normal_feature.max(dim=-1, keepdim=False)[0]


            
            x = torch.cat([x_all, x_type, x_para, normal_feature], dim=1)
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x))), self.drop)
            output_feats = self.mlp_seg_prob2(x).permute(0, 2, 1)  # 逐点特征






            
        coords_float = points.permute(0, 2, 1)


        # coords_float = torch.reshape(coords_float, (-1, coords_float.shape[-1]))
        coords_float = coords_float.reshape(-1, coords_float.shape[-1])
        if self.mode >0:
            coords_float = coords_float[:, 0:3]

        output_feats_puls_coods = torch.cat([x_all.permute(0, 2, 1), points.permute(0, 2, 1)], dim=-1)
        # pt_offsets = self.offset_linear(output_feats_puls_coods)

        output_feats_puls_coods = F.dropout(F.relu(self.bn3(self.conv3(output_feats_puls_coods.permute(0, 2, 1)))),
                                            self.drop)

        output_feats_puls_coods = output_feats_puls_coods.permute(0, 2, 1)

        semantic_scores = type_per_point_forgroup.view(-1, type_per_point_forgroup.shape[-1])




        if self.opt.ablation!=True:
            pt_offsets = self.offset_pred_block(points[:, 0:3, :].permute(0, 2, 1), output_feats_puls_coods,
                                                output_feats)

            pt_offsets = pt_offsets.permute(0, 2, 1)
            pt_offsets = torch.reshape(pt_offsets, (-1, pt_offsets.shape[-1]))
            # pt_offsets = pt_offsets.view(-1, pt_offsets.shape[-1])

        else:
            output_feats_puls_coods = torch.reshape(output_feats_puls_coods, (-1, output_feats_puls_coods.shape[-1]))
            pt_offsets = self.offset_linear(output_feats_puls_coods)



        proposals_idx, proposals_offset = self.forward_grouping(
            semantic_scores,
            pt_offsets,
            batch_idxs,
            coords_float,
            type_per_point,
            param_per_point,
            output_feats,
            lvl_fusion=False,
            training_mode='test'
            )

        # output_feats = torch.cat([output_feats, type_per_point_forgroup, param_per_point, points.permute(0, 2, 1)],
        #                          dim=-1)

        # output_feats = F.dropout(F.relu(
        #     self.bn_normal_squeeze_output_feature(self.mlp_squeeze_output_feature(output_feats.permute(0, 2, 1)))),
        #                          self.drop)
        # output_feats = output_feats.permute(0, 2, 1)

        # # output_feats_reshape = torch.reshape(output_feats, (-1, output_feats.shape[-1]))
        # output_feats_reshape = output_feats.view(-1, output_feats.shape[-1])

        output_feats_reshape = torch.reshape(output_feats, (-1, output_feats.shape[-1]))




        inst_feats, inst_map = self.clusters_voxelization(
            proposals_idx,
            proposals_offset,
            output_feats_reshape,
            coords_float,
            rand_quantize=False,
            spatial_shape=64,
            scale=64
        )
        instance_batch_idxs, cls_scores, iou_scores, mask_scores = self.forward_instance(
            inst_feats, inst_map)
        # print('instance_batch_idxs shape', instance_batch_idxs.shape)
        # print('cls_scores shape', cls_scores.shape)
        # print('iou_scores shape', iou_scores.shape)
        # print('mask_scores shape', mask_scores.shape)

        pred_instances = self.get_instances(
            proposals_idx,
            semantic_scores,
            cls_scores,
            iou_scores,
            mask_scores,
            v2p_map=v2p_maps,
            lvl_fusion=False,
            instance_classes=self.semantic_classes)

        if len(pred_instances) > 0:
            merge_instances = self.merge_masks(pred_instances)

            # semantic_pred=semantic_scores.max(1)[1]
            #
            # panoptic_preds = self.panoptic_fusion(semantic_pred.cpu().numpy(), pred_instances)


        else:
            merge_instances = np.zeros((pt_offsets.shape[0], 1), int)




        if self.mode==3:
            return type_per_point, param_per_point,normal_per_point, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
            , proposals_idx, proposals_offset, merge_instances, output_feats
        else:
            return type_per_point, param_per_point, \
                    semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                , proposals_idx, proposals_offset, merge_instances, output_feats




    def merge_masks(self, instances):
        """
        将多个形状为(N, 1)的点云实例mask合并成一个完整的实例标签。

        参数：
        masks: 一个列表，包含k个形状为(N, 1)的二进制数组，其中1表示该点属于该实例，0表示不属于该实例。

        返回：
        merged_mask: 形状为(N, 1)的整数数组，表示每个点的实例标签，实例标签为1到k之间的整数。
        """
        # 将所有mask合并成一个数组
        merge_instances = []
        for i in range(len(instances)):
            a_pred_instances = rle_decode(instances[i]['pred_mask'])
            merge_instances.append(a_pred_instances)
        merge_instances = np.array(merge_instances)
        # merged_mask = np.concatenate(merge_instances, axis=1)
        # 计算每个点的实例标签
        instance_labels = np.argmax(merge_instances, axis=0)
        # 将结果转换为形状为(N, 1)的整数数组
        merged_mask = instance_labels.reshape((-1, 1)).astype(np.int32)

        # merged_mask = instance_labels.view(-1, 1).astype(np.int32)

        return merged_mask

    def panoptic_fusion(self, semantic_preds, instance_preds):
        panoptic_skip_iou = 0.5
        cls_offset = 0
        panoptic_cls = semantic_preds.copy().astype(np.uint32)
        panoptic_ids = np.zeros_like(semantic_preds).astype(np.uint32)

        # higher score has higher fusion priority
        scores = [x['conf'] for x in instance_preds]
        score_inds = np.argsort(scores)[::-1]
        prev_paste = np.zeros_like(semantic_preds, dtype=bool)
        panoptic_id = 1
        for i in score_inds:
            instance = instance_preds[i]
            cls = instance['label_id']
            mask = rle_decode(instance['pred_mask']).astype(bool)

            # check overlap with pasted instances

            intersect = (mask * prev_paste).sum()
            if intersect / (mask.sum() + 1e-5) > panoptic_skip_iou:
                continue

            paste = mask * (~prev_paste)
            panoptic_cls[paste] = cls + cls_offset
            panoptic_ids[paste] = panoptic_id
            prev_paste[paste] = 1
            panoptic_id += 1

        # if thing classes have panoptic id == 0, ignore it
        ignore_inds = (panoptic_cls >= 11) & (panoptic_ids == 0)

        # encode panoptic results
        panoptic_preds = (panoptic_cls & 0xFFFF) | (panoptic_ids << 16)
        # panoptic_preds=panoptic_cls
        panoptic_preds[ignore_inds] = self.semantic_classes
        panoptic_preds = panoptic_preds.astype(np.uint32)
        return panoptic_preds

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self,
                      proposals_idx,
                      semantic_scores,
                      cls_scores,
                      iou_scores,
                      mask_scores,
                      v2p_map=None,
                      lvl_fusion=False,
                      instance_classes=10):

        cls_score_thr = 0.45  # 0.45
        mask_score_thr = -3 # -3
        sem2ins_classes = []
        min_npoint = 150  # 150

        if proposals_idx.size(0) == 0:
            return []

        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(instance_classes):
            if i in sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
                if lvl_fusion:
                    mask_pred = mask_pred[:, v2p_map.long()]
            else:

                cls_pred = cls_scores.new_full((num_instances,), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]  # class branch
                cur_iou_scores = iou_scores[:, i]  # mask score
                cur_mask_scores = mask_scores[:, i]  # seg branch 这个就是生成的mask
                score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
                mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')
                mask_inds = cur_mask_scores > mask_score_thr
                cur_proposals_idx = proposals_idx[mask_inds].long()
                # proposals_idx为某个分类的点的聚类编号（升序）和对应的idx（2）,proposals_offset为某个分类的多个聚类的每个类的点数（类数+1, 前面第一个是0）

                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

                # filter low score instance
                inds = cur_cls_scores > cls_score_thr
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
                # print('inds-a',mask_pred.shape)

                if lvl_fusion:
                    mask_pred = mask_pred[:, v2p_map.long()]

                # filter too small instances
                npoint = mask_pred.sum(1)
                # print('npoint',npoint)
                inds = npoint >= min_npoint
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
                # print('inds-b',mask_pred.shape[0])

            cls_pred_list.append(cls_pred.cpu())
            score_pred_list.append(score_pred.cpu())
            mask_pred_list.append(mask_pred.cpu())
        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()

        instances = []

        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)

        return instances

    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         type_per_point,
                         param_per_point,
                         feature_per_point,
                         lvl_fusion=False,
                         training_mode='train'
                         ):

        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        radius = 0.03  # 0.03
        similarity_threshold_inst = 0.989 # 0.94 ,0.995
        similarity_threshold_para = 0.00 # 0.97 0



        similarity_threshold_para_2 = {5: 0.8,
                                       1: 0.45,
                                       4: 0.45,
                                       3: 0.5,
                                       -1: 0.5,
                                       0: 0.5,
                                       2: 0.5,
                                       6: 0.5,
                                       7: 0.5,
                                       8: 0.5,
                                       9: 0.5}
        mean_active = 300  # 300
        npoint_thr = 0.15  # 0.15
        class_numpoint_mean = [532., 633., 812., 615., 776.,1408., 667., 629., 1054., 1911.]
        # class_numpoint_mean=[0., 0., 0., 0., 000000
        score_thr = 0.45  # 0.45
        with_pyramid = False
        with_octree = False
        base_size = 0.1
        min_npoint = 50  # 150,50
        class_numpoint_mean = torch.tensor(
            class_numpoint_mean, dtype=torch.float32)
        # assert class_numpoint_mean.size(0) == self.semantic_classes

        semantic_scores = \
            semantic_scores.view(type_per_point.shape[0], type_per_point.shape[1], -1)
        coords_float = coords_float.view(type_per_point.shape[0], type_per_point.shape[1], -1)
        pt_offsets = pt_offsets.view(type_per_point.shape[0], type_per_point.shape[1], -1)
        batch_idxs = batch_idxs.view(type_per_point.shape[0], type_per_point.shape[1], -1)

        proposals_idx_list = []
        proposals_offset_list = []

        for b in range(batch_size):
            semantic_scores_one = semantic_scores[b]
            batch_idxs_one = batch_idxs[b]
            coords_float_one = coords_float[b]
            pt_offsets_one = pt_offsets[b]
            # type_per_point_one = type_per_point[b]
            param_per_point_one = param_per_point[b]
            feature_per_point_one = feature_per_point[b]

            semantic_scores_one = semantic_scores_one.argmax(dim=1) #new

            for class_id in range(self.semantic_classes):
                torch.cuda.empty_cache()
                # 初始化空的张量

                # scores = semantic_scores_one[:, class_id].contiguous()
                # object_idxs = (scores > score_thr).nonzero().view(-1)



                object_idxs = (semantic_scores_one==class_id).nonzero().view(-1)#new



                if object_idxs.size(0) < min_npoint:
                    continue
                batch_idxs_ = batch_idxs_one[object_idxs]
                coords_ = coords_float_one[object_idxs]
                pt_offsets_ = pt_offsets_one[object_idxs]
                # type_per_point_ = type_per_point_one[object_idxs]
                param_per_point_ = param_per_point_one[object_idxs]
                feature_per_point_ = feature_per_point_one[object_idxs]

                batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)







                adj_mat_inst = compute_batch_adjacency_matrix(feature_per_point_, radius=0, dist_state=True)
                adj_mat_para = compute_batch_adjacency_matrix(param_per_point_,
                                                              radius=0, dist_state=True)

                adj_mat_para = adj_mat_para.squeeze(0)
                adj_mat_inst = adj_mat_inst.squeeze(0)


                # 通过batch for循环实现减少显存
                neighbor_inds, start_len = ball_query(
                    coords_ + pt_offsets_,
                    batch_idxs_,
                    batch_offsets_,
                    adj_mat_inst,
                    similarity_threshold_inst,
                    adj_mat_para,
                    similarity_threshold_para,
                    radius,
                    mean_active,
                    with_octree=with_octree)
                

                # proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, neighbor_inds.cpu(),
                #                                               start_len.cpu(), npoint_thr, class_id)
                

                
                semantic_preds_cpu = torch.ones(coords_float_one[object_idxs,0].shape)*class_id
                semantic_preds_cpu=semantic_preds_cpu.int().cpu()
                using_set_aggr_in_training = False
                using_set_aggr_in_testing = self.opt.using_set_aggr
                using_set_aggr = using_set_aggr_in_training if training_mode == 'train' else using_set_aggr_in_testing
                torch.cuda.empty_cache()
                proposals_idx, proposals_offset = hierarchical_aggregation(
                        semantic_preds_cpu, 
                        (coords_ + pt_offsets_).cpu(), 
                        neighbor_inds.cpu(), 
                        start_len.cpu(),
                        batch_idxs_.cpu(), 
                        training_mode, 
                        using_set_aggr)    
                
                
                
                
                
                # proposals_idx为某个分类的点的聚类编号（升序）和对应的idx（2）,proposals_offset为某个分类的多个聚类的每个类的点数（类数+1, 前面第一个是0）

                # del adj_mat_inst
                # del adj_mat_para
                # proposals_idx, proposals_offset = hierarchical_aggregation(
                #     semantic_preds_cpu, (coords_ + pt_offsets_).cpu(), neighbor_inds.cpu(), start_len.cpu(),
                #     batch_idxs_.cpu(), training_mode, using_set_aggr)

                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

                # merge proposals
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]
                if proposals_idx.size(0) > 0:
                    proposals_idx_list.append(proposals_idx)
                    proposals_offset_list.append(proposals_offset)
        if len(proposals_idx_list) > 0:
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
        else:
            proposals_idx = torch.zeros((0, 2), dtype=torch.int32)
            proposals_offset = torch.zeros((0,), dtype=torch.int32)

        #     proposals_idx_list_all.append(proposals_idx)
        #     proposals_offset_list_all.append(proposals_offset)
        #
        # proposals_idx_list_all = torch.cat(proposals_idx_list_all, dim=0)
        # proposals_offset_list_all = torch.cat(proposals_offset_list_all)

        return proposals_idx, proposals_offset \
 \


    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        if clusters_idx.size(0) == 0:
            # create dummpy tensors
            coords = torch.tensor(
                [[0, 0, 0, 0], [0, spatial_shape - 1, spatial_shape - 1, spatial_shape - 1]],
                dtype=torch.int,
                device='cuda')

            feats = feats[0:2]

            voxelization_feats = spconv.SparseConvTensor(feats, coords, [spatial_shape] * 3, 1)
            inp_map = feats.new_zeros((1,), dtype=torch.long)
            return voxelization_feats, inp_map

        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = sec_min(coords, clusters_offset.cuda())
        coords_max = sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1,), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    def forward_instance(self, inst_feats, inst_map):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores
        feats = self.global_pool(feats)
        cls_scores = self.cls_linear(feats)
        iou_scores = self.iou_score_linear(feats)
        return instance_batch_idxs, cls_scores, iou_scores, mask_scores


class PrimitiveNet(nn.Module):
    def __init__(self, opt):
        super(PrimitiveNet, self).__init__()
        self.opt = opt
        self.mode=self.opt.mode

        if self.opt.backbone == 'DGCNN':
            self.affinitynet = PrimitivesEmbeddingDGCNGn(
                opt=opt,
                emb_size=self.opt.out_dim,  
                num_primitives=self.opt.num_primitives,
                mode=self.opt.mode,
                num_channels=6,
            )

    def forward(self, xyz, normal,v2p_maps ,batch_idxs, postprocess=False):
        if self.mode==3:
            if postprocess:
                T_pred, T_param_pred, normal_per_point, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                    , proposals_idx, proposals_offset, pred_instances, output_feats \
                    = self.affinitynet(
                    xyz.transpose(1, 2).contiguous(),
                    normal.transpose(1, 2).contiguous(),
                    v2p_maps.contiguous(),
                    batch_idxs.contiguous(),
                    postprocess=postprocess)
                return T_pred, T_param_pred, normal_per_point, \
                        semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                        , proposals_idx, proposals_offset, pred_instances, output_feats
            else:
                T_pred, T_param_pred, normal_per_point, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                    , proposals_idx, proposals_offset, output_feats \
                    = self.affinitynet(
                    xyz.transpose(1, 2).contiguous(),
                    normal.transpose(1, 2).contiguous(),
                    v2p_maps.contiguous(),
                    batch_idxs.contiguous(),
                    postprocess=postprocess)

                return T_pred, T_param_pred,normal_per_point, \
                        semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                        , proposals_idx, proposals_offset, output_feats
        else:
            if postprocess:
                T_pred, T_param_pred, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                    , proposals_idx, proposals_offset, pred_instances, output_feats \
                    = self.affinitynet(
                    xyz.transpose(1, 2).contiguous(),
                    normal.transpose(1, 2).contiguous(),
                    v2p_maps.contiguous(),
                    batch_idxs.contiguous(),
                    postprocess=postprocess)
                return T_pred, T_param_pred, \
                        semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                        , proposals_idx, proposals_offset, pred_instances, output_feats
            else:
                T_pred, T_param_pred, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                    , proposals_idx, proposals_offset, output_feats \
                    = self.affinitynet(
                    xyz.transpose(1, 2).contiguous(),
                    normal.transpose(1, 2).contiguous(),
                    v2p_maps.contiguous(),
                    batch_idxs.contiguous(),
                    postprocess=postprocess)

                return T_pred, T_param_pred, \
                        semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
                        , proposals_idx, proposals_offset, output_feats









        # feat_spec_embedding, T_pred, normal_per_point, T_param_pred, subidx = self.affinitynet(
        #     xyz.transpose(1, 2).contiguous(),
        #     normal.transpose(1, 2).contiguous(),
        #     voxel_coords.contiguous(),
        #     p2v_maps.contiguous(),
        #     v2p_maps.contiguous(),
        #     spatial_shape,
        #     inds=inds,
        #     postprocess=postprocess)

        # if self.opt.input_normal:
        #     return feat_spec_embedding, T_pred, normal_per_point, T_param_pred, subidx
        # else:
        #     return feat_spec_embedding, T_pred, T_param_pred, subidx

        # if self.opt.input_normal:
        #     return T_pred, normal_per_point, T_param_pred, \
        # semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
        #     , proposals_idx, proposals_offset
        # else:
        #     return T_pred, T_param_pred, \
        # semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores \
        #     , proposals_idx, proposals_offset
