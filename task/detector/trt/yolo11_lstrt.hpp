#pragma once

#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#include "config_loshu.h"
#include "postprocess_loshu.h"

#include "../lo_shu_grid.hpp"
#include "preprocess.h"


namespace detector
{

class yolo11_lstrt
{
public:
  explicit yolo11_lstrt(const std::string & config_path, bool debug = false);
  ~yolo11_lstrt();

  std::optional<LoShuGrid> detect(
    const cv::Mat & raw_img,
    int frame_count = -1);

private:
  void load_engine();
  void release();

  std::optional<LoShuGrid> parse_trt_output(
    const float * decode_ptr_host,
    const cv::Mat & bgr_img,
    int frame_count);

  void draw(
    const cv::Mat & img,
    const LoShuGrid & grid,
    int frame_count) const;

  static cv::Rect clamp_rect_to_image(
    const cv::Rect & r,
    const cv::Mat & img);

private:
  CudaPreprocessContext preprocess_ctx_;
  bool debug_{false};

  std::string model_path_;
  std::string device_;

  float score_threshold_{0.25f};
  float nms_threshold_{0.45f};
  float kpt_threshold_{0.30f};

  size_t output_numel_{0};
  size_t output_bytes_{0};
  nvinfer1::DataType output_dtype_{nvinfer1::DataType::kFLOAT};

  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * context_{nullptr};

  cudaStream_t stream_{nullptr};

  float * input_buffer_device_{nullptr};
  float * output_buffer_device_{nullptr};

  float * decode_ptr_host_{nullptr};
  float * decode_ptr_device_{nullptr};

  int model_bboxes_{0};
};

}  // namespace detector
