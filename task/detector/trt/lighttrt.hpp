#pragma once

#include <list>
#include <string>

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#include "config_light.h"
#include "light.hpp"
#include "preprocess.h"

namespace detector
{

enum class LightTrtDecodeMode
{
  PackedPlugin,
  RawLTRB,
  OnnxXYWH
};

class LightTrt
{
public:
  explicit LightTrt(const std::string & config_path, bool debug = false);
  ~LightTrt();

  std::list<light> detect(const cv::Mat & raw_img, int frame_count = 0);

private:
  void load_engine();
  void release();

  std::list<light> parse_trt_output(
    const float * decode_ptr_host,
    const cv::Mat & bgr_img,
    int frame_count);

  bool check_name(const light & obj) const;
  bool check_confidence(const light & obj) const;

  lightname get_name(int class_id) const;

  void draw_detections(
    const cv::Mat & img,
    const std::list<light> & objs,
    int frame_count) const;

private:
  CudaPreprocessContext preprocess_ctx_;

  bool debug_{false};
  LightTrtDecodeMode decode_mode_{LightTrtDecodeMode::PackedPlugin};

  std::string model_path_;
  std::string device_;

  double min_confidence_{klightConfThresh};

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
