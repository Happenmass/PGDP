// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>   // 代替 THCDeviceUtils.cuh

#include <c10/cuda/CUDAGuard.h>           // 替代 THCState
#include <c10/cuda/CUDAStream.h>

#include <vector>
#include <iostream>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__host__ __device__ inline int64_t CeilDiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = CeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// boxes is a N x 5 tensor
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {
    using scalar_t = float;
    TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) >= 5,
                "boxes must have shape [N, >=5] (x1,y1,x2,y2,score,...)");

    // 绑定到与输入相同的设备，避免隐式切设备
    c10::cuda::CUDAGuard device_guard(boxes.get_device());

    // 根据 score 排序
    auto scores = boxes.select(1, 4);
    auto order_t = std::get<1>(scores.sort(/*dim=*/0, /*descending=*/true));
    auto boxes_sorted = boxes.index_select(0, order_t);

    const int boxes_num = static_cast<int>(boxes.size(0));
    if (boxes_num == 0) {
        return at::empty({0}, boxes.options().dtype(at::kLong).device(at::kCPU));
    }

    const int col_blocks = static_cast<int>(CeilDiv(boxes_num, threadsPerBlock));

    // 设备端数据指针（float）
    scalar_t* boxes_dev = boxes_sorted.data_ptr<scalar_t>();

    // 分配设备端掩码缓存
    unsigned long long* mask_dev = nullptr;
    C10_CUDA_CHECK(cudaMalloc(
        (void**)&mask_dev,
        static_cast<size_t>(boxes_num) * static_cast<size_t>(col_blocks) * sizeof(unsigned long long)
    ));

    // 启动 kernel
    dim3 blocks(
        static_cast<unsigned>(CeilDiv(boxes_num, threadsPerBlock)),
        static_cast<unsigned>(CeilDiv(boxes_num, threadsPerBlock))
    );
    dim3 threads(threadsPerBlock);

    nms_kernel<<<blocks, threads>>>(
        boxes_num,
        nms_overlap_thresh,
        boxes_dev,
        mask_dev
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 拷回掩码到主机
    std::vector<unsigned long long> mask_host(
        static_cast<size_t>(boxes_num) * static_cast<size_t>(col_blocks)
    );
    C10_CUDA_CHECK(cudaMemcpy(
        mask_host.data(),
        mask_dev,
        sizeof(unsigned long long) * mask_host.size(),
        cudaMemcpyDeviceToHost
    ));

    // 在 CPU 上执行选择逻辑
    std::vector<unsigned long long> remv(col_blocks);
    std::memset(remv.data(), 0, sizeof(unsigned long long) * col_blocks);

    at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
    int64_t* keep_out = keep.data_ptr<int64_t>();

    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; ++i) {
        const int nblock  = i / threadsPerBlock;
        const int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep_out[num_to_keep++] = i;
            const unsigned long long* p = mask_host.data() + static_cast<size_t>(i) * static_cast<size_t>(col_blocks);
            for (int j = nblock; j < col_blocks; ++j) {
                remv[j] |= p[j];
            }
        }
    }

    // 释放显存
    C10_CUDA_CHECK(cudaFree(mask_dev));

    // 按原始排序映射回到输入索引，并保持降序（与原实现一致）
    // order_t.index(keep[:num_to_keep]) -> 返回原 boxes 的索引；再 sort(0, false) 保持有序
    return std::get<0>(
        order_t.index({
            keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
                .to(order_t.device(), keep.scalar_type())
        }).sort(/*dim=*/0, /*descending=*/false)
    );
}