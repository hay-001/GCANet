/*
Ball Query with BatchIdx & Clustering Algorithm
Written by Li Jiang
All Rights Reserved 2020.
*/

#ifndef BFS_CLUSTER_H_EASY
#define BFS_CLUSTER_H_EASY
#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>

#include "../datatype/datatype.h"

int ballquery_batch_p_easy(at::Tensor xyz_tensor, at::Tensor batch_idxs_tensor,
                      at::Tensor batch_offsets_tensor, at::Tensor idx_tensor,
                      at::Tensor start_len_tensor, int n, int meanActive,
                      float radius);
int ballquery_batch_p_easy_cuda(int n, int meanActive, float radius,
                           const float *xyz, const int *batch_idxs,
                           const int *batch_offsets, int *idx, int *start_len,
                           cudaStream_t stream);


#endif // BFS_CLUSTER_H
