import os
import torch
import numpy as np
from trainer_new import Trainer
from option_new import build_option
from utils.loss_utils import compute_embedding_loss, compute_normal_loss, \
        compute_param_loss, compute_nnl_loss, compute_miou, compute_type_miou_abc\
    ,offset_loss,instance_loss
from utils.main_utils import npy
from utils.abc_utils import mean_shift, compute_entropy, construction_affinity_matrix_type, \
        construction_affinity_matrix_normal
import scipy.stats as stats
import random
import string
import time
import pandas as pd
from collections import Sequence


class MyTrainer(Trainer):

    def process_batch(self, batch_data_label, postprocess=False):
        # start=time.clock()

        inputs_xyz_th = (batch_data_label['gt_pc']).float().cuda().permute(0,2,1)
        inputs_n_th = (batch_data_label['gt_normal']).float().cuda().permute(0,2,1)
        inputs_v2p_map_th = (batch_data_label['v2p_map']).int().cuda()
        inputs_batch_idxs_th = (batch_data_label['batch_idx']).int().cuda()


        # if self.opt.input_normal:
        #     affinity_feat, type_per_point, normal_per_point, param_per_point, sub_idx = \
        #         self.model(inputs_xyz_th, inputs_n_th,inputs_voxel_coord_th,inputs_p2v_map_th,inputs_v2p_map_th,inputs_spatial_shape_th, postprocess=postprocess)
        # else:
        #     affinity_feat, type_per_point, param_per_point, sub_idx = \
        #         self.model(inputs_xyz_th, inputs_n_th,inputs_voxel_coord_th,inputs_p2v_map_th,inputs_v2p_map_th,inputs_spatial_shape_th, postprocess=postprocess)
        if self.opt.mode==3:
            if postprocess:
                type_per_point, param_per_point, normal_per_point, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores, proposals_idx, proposals_offset, pred_instances,affinity_feat \
                    = self.model(inputs_xyz_th, inputs_n_th, inputs_v2p_map_th, inputs_batch_idxs_th,
                                    postprocess=postprocess)

            else:
                type_per_point, param_per_point, normal_per_point, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores,proposals_idx, proposals_offset,affinity_feat\
                    = self.model(inputs_xyz_th, inputs_n_th,inputs_v2p_map_th,inputs_batch_idxs_th, postprocess=postprocess)
        else:
            if postprocess:

                type_per_point, param_per_point, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores, proposals_idx, proposals_offset, pred_instances,affinity_feat \
                    = self.model(inputs_xyz_th, inputs_n_th, inputs_v2p_map_th, inputs_batch_idxs_th,
                                    postprocess=postprocess)

            else:
                type_per_point, param_per_point, \
                semantic_scores, pt_offsets, instance_batch_idxs, cls_scores, iou_scores, mask_scores,proposals_idx, proposals_offset,affinity_feat\
                    = self.model(inputs_xyz_th, inputs_n_th,inputs_v2p_map_th,inputs_batch_idxs_th, postprocess=postprocess)


        

        # end=time.clock()
                    # print('feature extract time:',str(end-start))



        # sub_idx点的序号
        # inputs_xyz_sub = torch.gather(inputs_xyz_th, -1, sub_idx.unsqueeze(1).repeat(1,3,1))
        # N_gt = (batch_data_label['gt_normal']).float().cuda()
        # N_gt = torch.gather(N_gt, 1, sub_idx.unsqueeze(-1).repeat(1,1,3))
        # I_gt = torch.gather(batch_data_label['I_gt'], -1, sub_idx)#instance的gt
        # T_gt = torch.gather(batch_data_label['T_gt'], -1, sub_idx) #Segmentation的gt
        #

        # sub_idx点的序号
        inputs_xyz_sub = inputs_xyz_th
        N_gt = (batch_data_label['gt_normal']).float().cuda()
        I_gt = batch_data_label['I_gt']#instance的gt
        T_gt = batch_data_label['T_gt'] #Segmentation的gt

        instance_cls = batch_data_label['instance_cl'] #Segmentation的gt

        pt_offsets_gt = (batch_data_label['pt_offset_label']).float().cuda()
        instance_pointnum = (batch_data_label['instance_pointnum']).int().cuda()



        pt_offset_labels = torch.reshape(pt_offsets_gt, (pt_offsets_gt.shape[0] * pt_offsets_gt.shape[1], -1))
        instance_labels = torch.reshape(I_gt, (I_gt.shape[0] * I_gt.shape[1], -1)).long().cuda()


        loss_dict = {}

        # print(I_gt.shape)
        # print(affinity_feat.shape)
        # affinity_feat = torch.reshape(affinity_feat,(I_gt.shape[0],I_gt.shape[1],-1))
        # param_per_point = torch.reshape(param_per_point,(I_gt.shape[0],I_gt.shape[1],-1))
        # type_per_point = torch.reshape(type_per_point,(I_gt.shape[0],I_gt.shape[1],-1))



        if 'f' in self.opt.loss_class:#基元实例loss
            # network feature loss
            feat_loss, pull_loss, push_loss = compute_embedding_loss(affinity_feat, I_gt)
            loss_dict['feat_loss'] = feat_loss*2.0#1.0
        if self.opt.mode==3:#法向loss
            # normal angle loss
            normal_loss = compute_normal_loss(normal_per_point, N_gt)
            loss_dict['normal_loss'] = self.opt.normal_weight * normal_loss
        if 'p' in self.opt.loss_class:#参数loss
            # T_param_gt = torch.gather(batch_data_label['T_param'], 1, sub_idx.unsqueeze(-1).repeat(1,1,22))
            T_param_gt = batch_data_label['T_param']

            # parameter loss
            param_loss = compute_param_loss(param_per_point, T_gt, T_param_gt)
            loss_dict['param_loss'] = 5*self.opt.param_weight * param_loss
        if 'r' in self.opt.loss_class:#分割loss
            # primitive nnl loss
            type_loss = compute_nnl_loss(type_per_point, T_gt)
            loss_dict['nnl_loss'] = self.opt.type_weight * type_loss
            offset_loss_=10 * offset_loss(pt_offsets, instance_labels, pt_offset_labels)#10
            loss_dict['offset_loss'] = offset_loss_

            inst_loss=instance_loss(cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                              instance_labels, instance_pointnum, instance_cls, instance_batch_idxs, self.opt.num_primitives)
            loss_dict['inst_loss'] = inst_loss*1.0#0.1
        total_loss = 0

        for key in loss_dict:
            if 'loss' in key:
                if len(loss_dict[key].shape)>0:
                    total_loss += loss_dict[key][0]
                else:
                    total_loss += loss_dict[key]



        if postprocess:
            # type_per_point, normal_per_point, param_per_point,\
            # semantic_scores, pt_offsets, instance_batch_idxs, \
            # cls_scores, iou_scores, mask_scores,proposals_idx, proposals_offset\

            # print(type_per_point)
            # print(type_per_point.shape)
            # print(iou_scores.shape)
            # print(iou_scores)
            # print(instance_batch_idxs.shape)
            # print(instance_batch_idxs)
            # print(proposals_idx.shape)
            obj_idx = batch_data_label['index'][0]



             # print('mean_shift time:', str(end1 - start1))
            pred_instances = pred_instances.transpose([1,0])
            pred_instances = torch.tensor(pred_instances).to(torch.int64).to("cuda:0")

            # #只评估实例的分割效果
            miou = compute_miou(pred_instances, I_gt.to(torch.int64))
            loss_dict['miou'] = miou
# 评估实例的类别label的效果
            type_miou = compute_type_miou_abc(type_per_point, T_gt.to(torch.int64), pred_instances, I_gt.to(torch.int64))
            loss_dict['type_miou'] = type_miou
#
# #
#              end = time.clock()
             # print('compute loss time:', str(end - start))
#
#
            # print('ID:'+str(obj_idx)+'  ','miou:',round(miou.item(),3), 'type_miou:',round(type_miou.item(),3))
            torch.cuda.empty_cache()


#             # output_results = {'ID':obj_idx,
#             #                   'miou':round(miou.item(),3),
#             #                   'type_miou': round(type_miou.item(), 3),
#             #                   'feat_loss':round(feat_loss.item(),3),
#             #                   'param_loss': round(param_loss.item(),3),
#             #                   'nnl_loss': round(type_loss.item(),3),
#             #                   }
#             # df = pd.DataFrame()
#             # df = df.append(output_results, ignore_index=True)
#             # df.to_excel(os.path.join(ResultsSavePath,'output_results.xlsx'), index=True)
#
#             #########################################可视化
#
            if self.opt.resultsSave:

                visual_instance_offset=torch.cat([inputs_xyz_sub+pt_offsets.permute(1,0).unsqueeze(0),pred_instances.unsqueeze(0)], dim=1).squeeze(0).permute(1,0).data.cpu().numpy()
                visual_instance_offset_gt=torch.cat([inputs_xyz_sub+pt_offsets_gt.permute(0,2,1),I_gt.unsqueeze(0)], dim=1).squeeze(0).permute(1,0).data.cpu().numpy()

                visual_instance=torch.cat([inputs_xyz_sub,pred_instances.unsqueeze(0)], dim=1).squeeze(0).permute(1,0).data.cpu().numpy()

                type_per_point_label=torch.argmax(type_per_point, -1)
                visual_segmentic=torch.cat([inputs_xyz_sub,type_per_point_label.unsqueeze(0)], dim=1).squeeze(0).permute(1,0).data.cpu().numpy()

                visual_instance_gt=torch.cat([inputs_xyz_sub,I_gt.unsqueeze(0)], dim=1).squeeze(0).permute(1,0).data.cpu().numpy()

                visual_segmentic_gt=torch.cat([inputs_xyz_sub,T_gt.unsqueeze(0)], dim=1).squeeze(0).permute(1,0).data.cpu().numpy()

                # random_str = ''.join(random.sample(string.ascii_letters + string.digits, 20))
                if not os.path.exists(self.opt.log_dir+'/results/'):
                    os.makedirs(self.opt.log_dir+'/results/')


                np.savetxt(self.opt.log_dir+'/results/'+str(obj_idx)+'_miou:'+str(round(miou.item(),3))+ '_inc.xyz'
                                        , visual_instance, fmt='%.8f')
                np.savetxt(self.opt.log_dir+'/results/'+str(obj_idx)+'_typemiou:'+str(round(type_miou.item(),3))+ '_seg.xyz'
                                        , visual_segmentic, fmt='%.8f')
                np.savetxt(self.opt.log_dir+'/results/'+str(obj_idx)+'_miou:'+str(round(miou.item(),3))+ '_offset.xyz'
                                        , visual_instance_offset, fmt='%.8f')

                np.savetxt(self.opt.log_dir + '/results/'+str(obj_idx) + '_offset_gt.xyz'
                           , visual_instance_offset_gt, fmt='%.8f')
                np.savetxt(self.opt.log_dir + '/results/'+ str(obj_idx) + '_inc_gt.xyz'
                           , visual_instance_gt, fmt='%.8f')
                np.savetxt(self.opt.log_dir + '/results/'+ str(obj_idx) + '_seg_gt.xyz'
                           , visual_segmentic_gt, fmt='%.8f')

# # #########################################
#00
#
#
#
        return total_loss, loss_dict
        
if __name__=='__main__':
    FLAGS = build_option()
    trainer = MyTrainer(FLAGS)
    trainer.train()
#plane = 1; 圆锥cone = 3; 圆柱cylinder = 4; 球sphere = 5; open b-spline = 2, 8; close b-spline = 6, 7, 9.

#new    plane = 1; 圆锥cone = 3; 圆柱cylinder = 4; 球sphere = 5; open b-spline = 2; close b-spline = 6.



























