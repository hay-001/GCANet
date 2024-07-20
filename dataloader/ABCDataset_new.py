import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch
import numpy as np
from torch.utils import data
import h5py
import random
from collections import Counter
from src.augment_utils import rotate_perturbation_point_cloud, jitter_point_cloud, \
    shift_point_cloud, random_scale_point_cloud, rotate_point_cloud
# import pointgroup_ops
from softgroup.ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx,hierarchical_aggregation)
from option_new import build_option
from typing import Dict, Sequence, Tuple, Union

EPS = np.finfo(np.float32).eps

class ABCDataset(data.Dataset):
    def __init__(self, root, filename, opt, skip=1, fold=1, num_primitives=10):
        
        self.root = root
        self.data_path = open(os.path.join(root, filename), 'r')
        self.opt = opt
        self.augment_routines = [
            rotate_perturbation_point_cloud, jitter_point_cloud,
            shift_point_cloud, random_scale_point_cloud, rotate_point_cloud
        ]
        
        if 'train' in filename:
            self.augment = self.opt.augment
            self.if_normal_noise = self.opt.if_normal_noise
        else:
            self.augment = 0
            self.if_normal_noise = 0

       
        self.data_list = [item.strip() for item in self.data_path.readlines()]
        self.skip = skip
        
        self.data_list = self.data_list[::self.skip]
        self.tru_len = len(self.data_list)
        self.len = self.tru_len * fold
        self.num_primitives=num_primitives

    # def __getitem__(self, index):
    def __getitem__(self, index: int) -> Tuple:

        ret_dict = {}
        index = index % self.tru_len
        
        data_file = os.path.join(self.root, self.data_list[index] + '.h5')

        with h5py.File(data_file, 'r') as hf:


            points = np.array(hf.get("points"))#点云
            labels = np.array(hf.get("labels"))#基元实例
            normals = np.array(hf.get("normals"))#法向
            primitives = np.array(hf.get("prim"))#分割label
            primitive_param = np.array(hf.get("T_param")) #参数
        
        if self.augment:
            points = self.augment_routines[np.random.choice(np.arange(5))](points[None,:,:])[0]

        if self.if_normal_noise:
            scale=0.07
            noise = normals * np.clip(
                np.random.randn(points.shape[0], 1) * scale,
                a_min=-scale,
                a_max=scale)
            points = points + noise.astype(np.float32)
      
        ret_dict['gt_pc'] = points
        ret_dict['gt_normal'] = normals
        ret_dict['T_gt'] = primitives.astype(int)
        ret_dict['T_param'] = primitive_param
        
        # set small number primitive as background
        counter = Counter(labels)
        mapper = np.ones([labels.max() + 1]) * -1
        keys = [k for k, v in counter.items() if v > 100]
        if len(keys):
            mapper[keys] = np.arange(len(keys))
        label = mapper[labels]
        ret_dict['I_gt'] = label.astype(int)
        clean_primitives = np.ones_like(primitives) * -1
        valid_mask = label != -1
        clean_primitives[valid_mask] = primitives[valid_mask]

        if self.num_primitives==7:
            clean_primitives[clean_primitives == 7] = 6
            clean_primitives[clean_primitives == 9] = 6
            clean_primitives[clean_primitives == 8] = 2




        ret_dict['T_gt'] = clean_primitives.astype(int)

        ret_dict['index'] = self.data_list[index]

        small_idx = label == -1
        full_labels = label
        full_labels[small_idx] = labels[small_idx] + len\
            (keys)
        ret_dict['I_gt_clean'] = full_labels.astype(int)

        # l = np.arange(ret_dict['T_gt'].shape[0])
        #
        # np.random.shuffle(l)
        # random_index = torch.from_numpy(l[:7000])  # 随机采样7000个点



        #下采样
        subidx = np.random.choice(range(ret_dict['T_gt'].shape[0]), 7000, replace=False)
        ret_dict['gt_pc'] = ret_dict['gt_pc'][subidx]
        ret_dict['gt_normal'] = ret_dict['gt_normal'][subidx]
        ret_dict['T_gt'] = ret_dict['T_gt'][subidx]
        ret_dict['T_param'] = ret_dict['T_param'][subidx]
        ret_dict['I_gt'] = ret_dict['I_gt'][subidx]
        ret_dict['I_gt_clean'] = ret_dict['I_gt_clean'][subidx]


        inst_num, inst_pointnum, inst_cls, pt_offset_label\
            = self.getInstanceInfo(ret_dict['gt_pc'],
                                   ret_dict['I_gt'].astype(np.int32),
                                   ret_dict['T_gt'])





        ret_dict['inst_num'] = inst_num
        ret_dict['inst_pointnum'] = inst_pointnum
        ret_dict['inst_cls'] = inst_cls
        ret_dict['pt_offset_label'] = pt_offset_label

        # ret_dict['gt_pc'] = ret_dict['gt_pc']
        # ret_dict['gt_normal'] = ret_dict['gt_normal']
        # ret_dict['T_gt'] = ret_dict['T_gt']
        # ret_dict['T_param'] = ret_dict['T_param']
        # ret_dict['I_gt'] = ret_dict['I_gt']
        # ret_dict['I_gt_clean'] = ret_dict['I_gt_clean']



        return ret_dict




    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
        instance_pointnum = []
        instance_cls = []
        # max(instance_num, 0) to support instance_label with no valid instance_id
        instance_num = max(int(instance_label.max()) + 1, 0)
        # print(instance_label.min())

        for i_ in range(instance_num):
            # i_=i_-1
            inst_idx_i = np.where(instance_label == i_)
            xyz_i = xyz[inst_idx_i]

            pt_mean[inst_idx_i] = xyz_i.mean(0)
            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(semantic_label[cls_idx])
        pt_offset_label = pt_mean - xyz
        # print("pt_mean",pt_mean[0])
        # print("xyz",xyz[0])

        return instance_num, instance_pointnum, instance_cls, pt_offset_label



    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:

        gt_pcs, gt_normals, T_gts, T_params, I_gts, indexs, \
        I_gt_cleans, voxel_coords, v2p_maps, coords, \
        instance_pointnums, instance_clss, pt_offset_labels   \
            = [], [], [], [], [], [], [], [], [], [], [], [], []
        indexs=[]

        for i, data in enumerate(batch):

            gt_pcs.append(data['gt_pc'])
            gt_normals.append(data['gt_normal'])
            T_gts.append(data['T_gt'])
            T_params.append(data['T_param'])
            I_gts.append(data['I_gt'])
            indexs.append(data['index'])
            I_gt_cleans.append(data['I_gt_clean'])

            instance_pointnums.extend(data['inst_pointnum'])
            instance_clss.extend(data['inst_cls'])
            pt_offset_labels.append(data['pt_offset_label'])


            coord = torch.Tensor(data['gt_pc'])*128
            coord = coord.long()

            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))

        # T_gts = torch.cat(T_gts, 0)  # long (N)
        # instance_clss = torch.cat(instance_clss, 0).long()  # long (N)



        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), 128,None)  # long [3]
        # print('spatial_shape',spatial_shape,data['index'])
        # print(coords.max(0)[0][1:] + 1,data['index'])

        voxel_coords, v2p_maps, p2v_maps = voxelization_idx(coords, len(batch), 4)


        gt_pcs = torch.Tensor(np.array(gt_pcs))
        gt_normals = torch.Tensor(np.array(gt_normals))
        T_gts = torch.LongTensor(np.array(T_gts))
        T_params = torch.Tensor(np.array(T_params))
        I_gts = torch.IntTensor(np.array(I_gts))
        # indexs = torch.Tensor(indexs)
        I_gt_cleans = torch.IntTensor(np.array(I_gt_cleans))



        v2p_maps = torch.IntTensor(np.array(v2p_maps))
        voxel_coords = torch.LongTensor(np.array(voxel_coords))


        instance_pointnums = torch.LongTensor(np.array(instance_pointnums))
        instance_clss = torch.LongTensor(np.array(instance_clss))
        pt_offset_labels = torch.Tensor(np.array(pt_offset_labels))
        batch_idxs = coords[:, 0].int()


        # gt_pcs = torch.Tensor(gt_pcs)
        # gt_normals = torch.Tensor(gt_normals)
        # T_gts = torch.LongTensor(T_gts)
        # T_params = torch.Tensor(T_params)
        # I_gts = torch.IntTensor(I_gts)
        # # indexs = torch.Tensor(indexs)
        # I_gt_cleans = torch.IntTensor(I_gt_cleans)



        # p2v_maps = torch.IntTensor(p2v_maps)
        # v2p_maps = torch.IntTensor(v2p_maps)
        # voxel_coords = torch.LongTensor(voxel_coords)


        # instance_pointnums = torch.LongTensor(instance_pointnums)
        # instance_clss = torch.LongTensor(instance_clss)
        # pt_offset_labels = torch.Tensor(pt_offset_labels)
        # batch_idxs = coords[:, 0].int()




        # spatial_shape = torch.IntTensor(spatial_shape)


        # voxel_coords = torch.cat(voxel_coords, 0)  # float [B*N, 3]
        # p2v_maps = torch.cat(p2v_maps, 0)
        # v2p_maps = torch.cat(v2p_maps, 0)
        # voxel_coords = np.array(voxel_coords)
        # voxel_coords = np.array(voxel_coords)


        # p2v_maps = torch.Tensor(p2v_maps)
        # v2p_maps = torch.Tensor(v2p_maps)
        # voxel_coords = torch.Tensor(voxel_coords)


        return {
            'gt_pc': gt_pcs,
            'gt_normal': gt_normals,
            'T_gt': T_gts,
            'T_param': T_params,
            'I_gt': I_gts,
            'index': indexs,
            'I_gt_clean': I_gt_cleans,
            'voxel_coord': voxel_coords,
            'v2p_map': v2p_maps,
            'instance_pointnum': instance_pointnums,
            'instance_cl': instance_clss,
            'pt_offset_label': pt_offset_labels,
            'batch_idx': batch_idxs
        }




    def __len__(self):
        return self.len

if __name__ == '__main__':
    FLAGS = build_option()
    abc_dataset = ABCDataset('./data/ABC/',
                            '00000.h5',
                             opt=FLAGS
                             )

    for idx in range(len(abc_dataset)):
        example = abc_dataset[idx]
        import ipdb
        ipdb.set_trace()
