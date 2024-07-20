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
import pointgroup_ops
from option import build_option
from typing import Dict, Sequence, Tuple, Union

EPS = np.finfo(np.float32).eps

class ABCDataset(data.Dataset):
    def __init__(self, root, filename, opt, skip=1, fold=1):
        
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
            noise = normals * np.clip(
                np.random.randn(points.shape[0], 1) * 0.01,
                a_min=-0.01,
                a_max=0.01)
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




        # ret_dict['gt_pc'] = ret_dict['gt_pc']
        # ret_dict['gt_normal'] = ret_dict['gt_normal']
        # ret_dict['T_gt'] = ret_dict['T_gt']
        # ret_dict['T_param'] = ret_dict['T_param']
        # ret_dict['I_gt'] = ret_dict['I_gt']
        # ret_dict['I_gt_clean'] = ret_dict['I_gt_clean']





        return ret_dict



    # def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
    #     # gt_pcs, gt_normals, T_gts, T_params, I_gts, indexs, I_gt_cleans, voxel_coords, p2v_maps, v2p_maps,   \
    #     #     = [], [], [], [], [], [], [], [], [], []
    #     indexs=[]
    #     for i, data in enumerate(batch):
    #
    #         # gt_pcs.append(data['gt_pc'])
    #         # gt_normals.append(data['gt_normal'])
    #         # T_gts.append(data['T_gt'])
    #         # T_params.append(data['T_param'])
    #         # I_gts.append(data['I_gt'])
    #         # indexs.append(data['index'])
    #         # I_gt_cleans.append(data['I_gt_clean'])
    #
    #         coord = torch.Tensor(data['gt_pc'])*1000000
    #         coord = coord.long()
    #         coord = torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(1), coord], 1)
    #
    #         # a, b, c = pointgroup_ops.voxelization_idx(coords, len(batch), 4)
    #         indexs.append(data['index'])
    #         if i== 0:
    #             gt_pcs=torch.unsqueeze(torch.Tensor(data['gt_pc']),0)
    #             gt_normals=torch.unsqueeze(torch.Tensor(data['gt_normal']),0)
    #             T_gts=torch.unsqueeze(torch.LongTensor(data['T_gt']),0)
    #             T_params=torch.unsqueeze(torch.Tensor(data['T_param']),0)
    #             I_gts=torch.unsqueeze(torch.IntTensor(data['I_gt']),0)
    #             # indexs=torch.unsqueeze(data['index'],0)
    #             I_gt_cleans=torch.unsqueeze(torch.IntTensor(data['I_gt_clean']),0)
    #
    #             spatial_shape = np.clip((torch.Tensor(data['gt_pc']).max(0)[0][1:] + 1).numpy(), 128,
    #                                     None)  # long [3]
    #             spatial_shapes=torch.unsqueeze(torch.IntTensor(spatial_shape),0)
    #
    #
    #             voxel_coord, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coord, len(batch), 4)
    #             voxel_coords=torch.unsqueeze(voxel_coord,0)
    #             p2v_maps=torch.unsqueeze(p2v_map,0)
    #             v2p_maps=torch.unsqueeze(v2p_map,0)
    #
    #         else:
    #             gt_pcs=torch.cat([gt_pcs,torch.unsqueeze(torch.Tensor(data['gt_pc']),0)], 0)
    #             gt_normals=torch.cat([gt_normals,torch.unsqueeze(torch.Tensor(data['gt_normal']),0)], 0)
    #             T_gts=torch.cat([T_gts,torch.unsqueeze(torch.LongTensor(data['T_gt']),0)], 0)
    #             T_params=torch.cat([T_params,torch.unsqueeze(torch.Tensor(data['T_param']),0)], 0)
    #             I_gts=torch.cat([I_gts,torch.unsqueeze(torch.IntTensor(data['I_gt']),0)], 0)
    #             # indexs=torch.cat([indexs,torch.unsqueeze(data['indexs'],0)], 0)
    #             I_gt_cleans=torch.cat([I_gt_cleans,torch.unsqueeze(torch.IntTensor(data['I_gt_clean']),0)], 0)
    #
    #             spatial_shape = np.clip((torch.Tensor(data['gt_pc']).max(0)[0][1:] + 1).numpy(), 128,
    #                                     None)  # long [3]
    #             spatial_shapes=torch.cat([spatial_shapes,torch.unsqueeze(torch.IntTensor(spatial_shape),0)], 0)
    #
    #
    #             voxel_coord, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coord, len(batch), 4)
    #             voxel_coords=torch.cat([voxel_coords,torch.unsqueeze(voxel_coord,0)], 0)
    #             p2v_maps=torch.cat([p2v_maps,torch.unsqueeze(p2v_map,0)], 0)
    #             v2p_maps=torch.cat([v2p_maps,torch.unsqueeze(v2p_map,0)], 0)
    #             print(voxel_coord, p2v_map, v2p_map)
    #
    #
    #         # voxel_coord, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coord, len(batch), 4)
    #         #
    #         # print(p2v_map)
    #         # voxel_coords.append(voxel_coord)
    #         # p2v_maps.append(p2v_map)
    #         # v2p_maps.append(v2p_map)
    #
    #
    #
    #     # gt_pcs = torch.Tensor(gt_pcs)
    #     # gt_normals = torch.Tensor(gt_normals)
    #     # T_gts = torch.Tensor(T_gts)
    #     # T_params = torch.Tensor(T_params)
    #     # I_gts = torch.Tensor(I_gts)
    #     # # indexs = torch.Tensor(indexs)
    #     # I_gt_cleans = torch.Tensor(I_gt_cleans)
    #
        # # voxel_coords = torch.cat(voxel_coords, 0)  # float [B*N, 3]
        # # p2v_maps = torch.cat(p2v_maps, 0)
        # # v2p_maps = torch.cat(v2p_maps, 0)
        # # voxel_coords = np.array(voxel_coords)
        # # voxel_coords = np.array(voxel_coords)
        #
        #
        # # p2v_maps = torch.Tensor(p2v_maps)
        # # v2p_maps = torch.Tensor(v2p_maps)
        # # voxel_coords = torch.Tensor(voxel_coords)
        #
        #
        # return {
        #     'gt_pc': gt_pcs,
        #     'gt_normal': gt_normals,
        #     'T_gt': T_gts,
        #     'T_param': T_params,
        #     'I_gt': I_gts,
        #     'index': indexs,
        #     'I_gt_clean': I_gt_cleans,
        #     'voxel_coord': voxel_coords,
        #     'p2v_map': p2v_maps,
        #     'v2p_map': v2p_maps,
        #     'spatial_shape': spatial_shapes
        #
        # }

    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        gt_pcs, gt_normals, T_gts, T_params, I_gts, indexs, I_gt_cleans, voxel_coords, p2v_maps, v2p_maps, coords  \
            = [], [], [], [], [], [], [], [], [], [], []
        indexs=[]

        for i, data in enumerate(batch):

            gt_pcs.append(data['gt_pc'])
            gt_normals.append(data['gt_normal'])
            T_gts.append(data['T_gt'])
            T_params.append(data['T_param'])
            I_gts.append(data['I_gt'])
            indexs.append(data['index'])
            I_gt_cleans.append(data['I_gt_clean'])

            coord = np.round(data['gt_pc']*64).astype(np.int64)  # 将坐标转换为长整型
            coord = torch.LongTensor(coord)

            # coord = torch.Tensor(data['gt_pc'])*20
            # coord = coord.long()

            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), 64,None)  # long [3]

        voxel_coords, v2p_maps, p2v_maps = pointgroup_ops.voxelization_idx(coords, len(batch), 4)


        gt_pcs = torch.Tensor(gt_pcs)
        gt_normals = torch.Tensor(gt_normals)
        T_gts = torch.LongTensor(T_gts)
        T_params = torch.Tensor(T_params)
        I_gts = torch.IntTensor(I_gts)
        # indexs = torch.Tensor(indexs)
        I_gt_cleans = torch.IntTensor(I_gt_cleans)



        p2v_maps = torch.IntTensor(p2v_maps)
        v2p_maps = torch.IntTensor(v2p_maps)
        voxel_coords = torch.LongTensor(voxel_coords)
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
            'p2v_map': p2v_maps,
            'v2p_map': v2p_maps,
            'spatial_shape': spatial_shape

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
