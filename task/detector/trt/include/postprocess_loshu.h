#pragma once

#include <cuda_runtime.h>

void cuda_decode_loshu_onnx_xywh(
    const float* predict,
    int num_bboxes,
    float confidence_threshold,
    float* parray,
    int max_objects,
    cudaStream_t stream);

void cuda_nms_loshu(
    float* parray,
    float nms_threshold,
    int max_objects,
    cudaStream_t stream);

