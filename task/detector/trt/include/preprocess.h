#pragma once

#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "types.h"

// struct AffineMatrix
// {
//   float value[6];
// };

struct CudaPreprocessContext
{
  uint8_t* img_buffer_host = nullptr;
  uint8_t* img_buffer_device = nullptr;
  int max_image_size = 0;
};

void cuda_preprocess_init(
  CudaPreprocessContext& ctx,
  int max_image_size);

void cuda_preprocess_destroy(
  CudaPreprocessContext& ctx);

void cuda_preprocess(
  CudaPreprocessContext& ctx,
  uint8_t* src,
  int src_width,
  int src_height,
  float* dst,
  int dst_width,
  int dst_height,
  cudaStream_t stream);

void cuda_batch_preprocess(
  CudaPreprocessContext& ctx,
  std::vector<cv::Mat>& img_batch,
  float* dst,
  int dst_width,
  int dst_height,
  cudaStream_t stream);
