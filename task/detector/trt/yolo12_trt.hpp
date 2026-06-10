#pragma once

#include <list>
#include <string>

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#include "config.h"
#include "types.h"

#include "../KFS.hpp"
#include "preprocess.h"


enum class TrtDecodeMode
{
  PackedPlugin,   // [count][Detection...]
  RawLTRB,        // ltrb + cls logits
  OnnxXYWH        // xywh + cls probs
};

namespace detector
{

class yolo12_trt
{
public:
  explicit yolo12_trt(const std::string & config_path, bool debug = false);
  ~yolo12_trt();

  std::list<KFS> detect(const cv::Mat & raw_img, int frame_count = 0);
  void save(const KFS & kfs) const;

private:
  void load_engine();
  void release();

  // GPU decode + NMS 后的 host 结果解析
  std::list<KFS> parse_trt_output(
    const float * decode_ptr_host,
    const cv::Mat & bgr_img,
    int frame_count);

  bool check_name(const KFS & kfs) const;
  bool check_confidence(const KFS & kfs) const;

  cv::Point2f get_center(const cv::Rect & box) const;
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  KFSname get_name(int class_id) const;
  Colors get_color(const cv::Mat & pattern) const;

  void draw_detections(
    const cv::Mat & img,
    const std::list<KFS> & kfss,
    int frame_count) const;

private:
  CudaPreprocessContext preprocess_ctx_;
  bool debug_ = false;
  bool use_roi_ = false;
  TrtDecodeMode decode_mode_{TrtDecodeMode::PackedPlugin};

  std::string model_path_;
  std::string device_;
  std::string save_path_;

  double min_confidence_ = kConfThresh;

  cv::Rect roi_{0, 0, 0, 0};
  cv::Point2f offset_{0.0f, 0.0f};

  cv::Mat tmp_img_;

  size_t output_numel_{0};
  size_t output_bytes_{0};
  nvinfer1::DataType output_dtype_{nvinfer1::DataType::kFLOAT};
  nvinfer1::IRuntime * runtime_ = nullptr;
  nvinfer1::ICudaEngine * engine_ = nullptr;
  nvinfer1::IExecutionContext * context_ = nullptr;

  cudaStream_t stream_ = nullptr;

  // =========================
  // Device / Host Buffer
  // =========================
  // 输入：1 x 3 x H x W
  float * input_buffer_device_ = nullptr;

  // TRT 原始输出 buffer（device）
  float * output_buffer_device_ = nullptr;

  // GPU decode + NMS 输出
  float * decode_ptr_host_ = nullptr;
  float * decode_ptr_device_ = nullptr;

  // 模型原始输出中的 bbox 数量（由 engine 输出 shape 推出来）
  int model_bboxes_ = 0;
};

}  // namespace detector
