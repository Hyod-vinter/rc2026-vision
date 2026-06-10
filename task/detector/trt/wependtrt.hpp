#pragma once

#include <list>
#include <string>

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#include "config_wepend.h"
#include "wepend.hpp"
#include "preprocess.h"


namespace detector
{

enum class WependTrtDecodeMode
{
  PackedPlugin,
  RawLTRB,
  OnnxXYWH
};

class WependTrt
{
public:
  explicit WependTrt(const std::string & config_path, bool debug = false);
  ~WependTrt();

  std::list<WependObject> detect(const cv::Mat & raw_img, int frame_count = 0);

private:
  CudaPreprocessContext preprocess_ctx_;
  void load_engine();
  void release();

  std::list<WependObject> parse_trt_output(
    const float * decode_ptr_host,
    const cv::Mat & bgr_img,
    int frame_count);

  bool check_name(const WependObject & obj) const;
  bool check_confidence(const WependObject & obj) const;

  cv::Point2f get_center(const cv::Rect & box) const;
  cv::Point2f get_image_center(const cv::Mat & img) const;

  std::string get_name(int class_id) const;

  void draw_detections(
    const cv::Mat & img,
    const std::list<WependObject> & objs,
    int frame_count) const;

private:
  bool debug_{false};
  WependTrtDecodeMode decode_mode_{WependTrtDecodeMode::PackedPlugin};

  std::string model_path_;
  std::string device_;

  double min_confidence_{kwependConfThresh};

  cv::Mat tmp_img_;

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