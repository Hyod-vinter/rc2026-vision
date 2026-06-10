#include "postprocess_loshu.h"

#include <cuda_runtime.h>
#include <cmath>

namespace
{
constexpr int kLoShuKptNumCuda = 10;
constexpr int kLoShuElementCuda = 7 + kLoShuKptNumCuda * 3;

static __device__ float loshu_box_iou(
    float aleft,
    float atop,
    float aright,
    float abottom,
    float bleft,
    float btop,
    float bright,
    float bbottom)
{
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);

    if (c_area <= 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);

    return c_area / (a_area + b_area - c_area + 1e-6f);
}

static __global__ void decode_kernel_loshu_onnx_xywh(
    const float* predict,
    int num_bboxes,
    float confidence_threshold,
    float* parray,
    int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;

    if (position >= num_bboxes)
        return;

    // Engine output confirmed:
    // output0 = [1, 35, 8400]
    // 35 = x,y,w,h,conf + 10 * (kx,ky,ks)
    float cx = predict[0 * num_bboxes + position];
    float cy = predict[1 * num_bboxes + position];
    float w  = predict[2 * num_bboxes + position];
    float h  = predict[3 * num_bboxes + position];
    float confidence = predict[4 * num_bboxes + position];

    if (confidence < confidence_threshold)
        return;

    int index = static_cast<int>(atomicAdd(parray, 1.0f));

    if (index >= max_objects)
        return;

    float left   = cx - 0.5f * w;
    float top    = cy - 0.5f * h;
    float right  = cx + 0.5f * w;
    float bottom = cy + 0.5f * h;

    float* pout_item = parray + 1 + index * kLoShuElementCuda;

    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = 0.0f;  // class_id, LoShu grid is single-class
    *pout_item++ = 1.0f;  // keep flag

    #pragma unroll
    for (int k = 0; k < kLoShuKptNumCuda; ++k) {
        int base = 5 + k * 3;

        float kx = predict[(base + 0) * num_bboxes + position];
        float ky = predict[(base + 1) * num_bboxes + position];
        float ks = predict[(base + 2) * num_bboxes + position];

        *pout_item++ = kx;
        *pout_item++ = ky;
        *pout_item++ = ks;
    }
}

static __global__ void nms_kernel_loshu(
    float* bboxes,
    int max_objects,
    float threshold)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    int count = min(static_cast<int>(bboxes[0]), max_objects);

    if (position >= count)
        return;

    float* pcurrent = bboxes + 1 + position * kLoShuElementCuda;

    for (int i = 0; i < count; ++i) {
        if (i == position)
            continue;

        float* pitem = bboxes + 1 + i * kLoShuElementCuda;

        // same class only. LoShu is single-class, class id is still kept at [5].
        if (pcurrent[5] != pitem[5])
            continue;

        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = loshu_box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold) {
                pcurrent[6] = 0.0f;
                return;
            }
        }
    }
}
}  // namespace

void cuda_decode_loshu_onnx_xywh(
    const float* predict,
    int num_bboxes,
    float confidence_threshold,
    float* parray,
    int max_objects,
    cudaStream_t stream)
{
    int block = 256;
    int grid = static_cast<int>(ceil(num_bboxes / static_cast<float>(block)));

    decode_kernel_loshu_onnx_xywh<<<grid, block, 0, stream>>>(
        predict,
        num_bboxes,
        confidence_threshold,
        parray,
        max_objects);
}

void cuda_nms_loshu(
    float* parray,
    float nms_threshold,
    int max_objects,
    cudaStream_t stream)
{
    int block = max_objects < 256 ? max_objects : 256;
    int grid = static_cast<int>(ceil(max_objects / static_cast<float>(block)));

    nms_kernel_loshu<<<grid, block, 0, stream>>>(
        parray,
        max_objects,
        nms_threshold);
}
