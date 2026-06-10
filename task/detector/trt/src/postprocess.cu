//
// Created by lindsay on 23-7-17.
//
#include "postprocess.h"
#include "types.h"

static __device__ float sigmoid_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __global__ void decode_kernel_obb(float* predict, int num_bboxes, float confidence_threshold, float* parray,
                                         int max_objects) {
    float count = predict[0];
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= count)
        return;

    float* pitem = predict + 1 + position * (sizeof(Detection) / sizeof(float));
    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float confidence = pitem[4];

    if (confidence < confidence_threshold)
        return;
    //[center_x center_y w h conf class_id  mask[32] keypoints[51] angle]
    float cx = pitem[0];
    float cy = pitem[1];
    float width = pitem[2];
    float height = pitem[3];
    float label = pitem[5];
    float angle = pitem[89];

    float* pout_item = parray + 1 + index * bbox_element;
    *pout_item++ = cx;
    *pout_item++ = cy;
    *pout_item++ = width;
    *pout_item++ = height;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = angle;
}

static __global__ void decode_kernel(float* predict, int num_bboxes, float confidence_threshold, float* parray,
                                     int max_objects) {
    float count = predict[0];
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= count)
        return;

    float* pitem = predict + 1 + position * (sizeof(Detection) / sizeof(float));
    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float confidence = pitem[4];
    if (confidence < confidence_threshold)
        return;

    float left = pitem[0];
    float top = pitem[1];
    float right = pitem[2];
    float bottom = pitem[3];
    float label = pitem[5];

    float* pout_item = parray + 1 + index * bbox_element;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
}

static __global__ void decode_kernel_raw_head(
    const float* predict,
    int num_bboxes,
    int input_w,
    int input_h,
    int num_classes,
    float confidence_threshold,
    float* parray,
    int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // 这版针对 YOLO12 det raw head:
    // shape = [1, 4 + num_classes, total_boxes]
    // layout = [attrs, boxes]
    // 前 4 个通道是 l,t,r,b 距离，后面是 cls logits
    // 三个尺度默认是 stride 8 / 16 / 32
    const int strides[3] = {8, 16, 32};

    int stride = 0;
    int grid_w = 0;
    int grid_h = 0;
    int local_idx = position;

    int base = 0;
    bool found = false;

    #pragma unroll
    for (int si = 0; si < 3; ++si) {
        int gw = input_w / strides[si];
        int gh = input_h / strides[si];
        int count = gw * gh;

        if (position < base + count) {
            stride = strides[si];
            grid_w = gw;
            grid_h = gh;
            local_idx = position - base;
            found = true;
            break;
        }
        base += count;
    }

    if (!found)
        return;

    int row = local_idx / grid_w;
    int col = local_idx % grid_w;

    int class_id = -1;
    float max_cls_prob = 0.0f;

    // 类别分数要做 sigmoid，和 plugin 的 CalDetection 保持一致
    for (int c = 0; c < num_classes; ++c) {
        float p = sigmoid_device(predict[(4 + c) * num_bboxes + position]);
        if (p > max_cls_prob) {
            max_cls_prob = p;
            class_id = c;
        }
    }

    if (max_cls_prob < confidence_threshold)
        return;

    // 前 4 个是 l,t,r,b
    float l = predict[0 * num_bboxes + position];
    float t = predict[1 * num_bboxes + position];
    float r = predict[2 * num_bboxes + position];
    float b = predict[3 * num_bboxes + position];

    // 完全对齐你上传的 YoloLayerPlugin::CalDetection()
    float left   = (col + 0.5f - l) * stride;
    float top    = (row + 0.5f - t) * stride;
    float right  = (col + 0.5f + r) * stride;
    float bottom = (row + 0.5f + b) * stride;

    int index = static_cast<int>(atomicAdd(parray, 1.0f));
    if (index >= max_objects)
        return;

    float* pout_item = parray + 1 + index * bbox_element;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = max_cls_prob;
    *pout_item++ = static_cast<float>(class_id);
    *pout_item++ = 1.0f;  // keep
}

static __global__ void decode_kernel_onnx_xywh(
    const float* predict,
    int num_bboxes,
    int num_classes,
    float confidence_threshold,
    float* parray,
    int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // ONNX + trtexe 输出假设：
    // shape = [1, 4 + num_classes, num_bboxes]
    // layout = [attrs, boxes]
    // 前4个已经是 cx, cy, w, h（网络输入尺度下）
    // 后面的类别已经是概率，不再做 sigmoid

    float cx = predict[0 * num_bboxes + position];
    float cy = predict[1 * num_bboxes + position];
    float w  = predict[2 * num_bboxes + position];
    float h  = predict[3 * num_bboxes + position];

    int class_id = -1;
    float confidence = 0.0f;

    for (int c = 0; c < num_classes; ++c) {
        float p = predict[(4 + c) * num_bboxes + position];  // 不做 sigmoid
        if (p > confidence) {
            confidence = p;
            class_id = c;
        }
    }

    if (confidence < confidence_threshold)
        return;

    float left   = cx - 0.5f * w;
    float top    = cy - 0.5f * h;
    float right  = cx + 0.5f * w;
    float bottom = cy + 0.5f * h;

    int index = static_cast<int>(atomicAdd(parray, 1.0f));
    if (index >= max_objects)
        return;

    float* pout_item = parray + 1 + index * bbox_element;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = static_cast<float>(class_id);
    *pout_item++ = 1.0f;
}


static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop,
                                float bright, float bbottom) {
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min(static_cast<int>(bboxes[0]), max_objects);
    if (position >= count)
        return;

    float* pcurrent = bboxes + 1 + position * bbox_element;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * bbox_element;
        if (i == position || pcurrent[5] != pitem[5])
            continue;
        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;
            float iou =
                    box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);
            if (iou > threshold) {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

static __device__ void convariance_matrix(float w, float h, float r, float& a, float& b, float& c) {
    float a_val = w * w / 12.0f;
    float b_val = h * h / 12.0f;
    float cos_r = cosf(r);
    float sin_r = sinf(r);

    a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
    b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
    c = (a_val - b_val) * sin_r * cos_r;
}

static __device__ float box_probiou(float cx1, float cy1, float w1, float h1, float r1, float cx2, float cy2, float w2,
                                    float h2, float r2, float eps = 1e-7) {

    // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    float a1, b1, c1, a2, b2, c2;
    convariance_matrix(w1, h1, r1, a1, b1, c1);
    convariance_matrix(w2, h2, r2, a2, b2, c2);

    float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) /
               ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t3 = logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2)) /
                            (4 * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps) +
                    eps);
    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd = fmaxf(fminf(bd, 100.0f), eps);
    float hd = sqrtf(1.0f - expf(-bd) + eps);
    return 1 - hd;
}

static __global__ void nms_kernel_obb(float* bboxes, int max_objects, float threshold) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min(static_cast<int>(bboxes[0]), max_objects);
    if (position >= count)
        return;

    float* pcurrent = bboxes + 1 + position * bbox_element;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * bbox_element;
        if (i == position || pcurrent[5] != pitem[5])
            continue;
        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;
            float iou = box_probiou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pcurrent[7], pitem[0], pitem[1],
                                    pitem[2], pitem[3], pitem[7]);
            if (iou > threshold) {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

void cuda_decode(float* predict, int num_bboxes, float confidence_threshold, float* parray, int max_objects,
                 cudaStream_t stream) {
    int block = 256;
    int grid = ceil(num_bboxes / (float)block);
    decode_kernel<<<grid, block, 0, stream>>>((float*)predict, num_bboxes, confidence_threshold, parray, max_objects);
}

void cuda_nms(float* parray, float nms_threshold, int max_objects, cudaStream_t stream) {
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}

void cuda_decode_raw_head(
    const float* predict,
    int num_bboxes,
    int input_w,
    int input_h,
    int num_classes,
    float confidence_threshold,
    float* parray,
    int max_objects,
    cudaStream_t stream)
{
    int block = 256;
    int grid = static_cast<int>(ceil(num_bboxes / (float)block));

    decode_kernel_raw_head<<<grid, block, 0, stream>>>(
        predict,
        num_bboxes,
        input_w,
        input_h,
        num_classes,
        confidence_threshold,
        parray,
        max_objects);
}

void cuda_decode_onnx_xywh(
    const float* predict,
    int num_bboxes,
    int num_classes,
    float confidence_threshold,
    float* parray,
    int max_objects,
    cudaStream_t stream)
{
    int block = 256;
    int grid = static_cast<int>(ceil(num_bboxes / (float)block));

    decode_kernel_onnx_xywh<<<grid, block, 0, stream>>>(
        predict,
        num_bboxes,
        num_classes,
        confidence_threshold,
        parray,
        max_objects);
}


void cuda_decode_obb(float* predict, int num_bboxes, float confidence_threshold, float* parray, int max_objects,
                     cudaStream_t stream) {
    int block = 256;
    int grid = ceil(num_bboxes / (float)block);
    decode_kernel_obb<<<grid, block, 0, stream>>>((float*)predict, num_bboxes, confidence_threshold, parray,
                                                  max_objects);
}

void cuda_nms_obb(float* parray, float nms_threshold, int max_objects, cudaStream_t stream) {
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    nms_kernel_obb<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}
