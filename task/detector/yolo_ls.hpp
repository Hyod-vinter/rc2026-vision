#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include <optional>
#include <string>
#include <vector>

#include "lo_shu_grid.hpp"

namespace detector
{

class YOLOLS
{
public:
  YOLOLS(const std::string & config_path, bool debug = false);

  std::optional<LoShuGrid> detect(
    const cv::Mat & raw_img,
    int frame_count = -1);

private:
  std::optional<LoShuGrid> parse(
    double scale,
    cv::Mat & output,
    const cv::Mat & bgr_img,
    int frame_count);

  void draw(
    const cv::Mat & img,
    const LoShuGrid & grid,
    int frame_count) const;

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;

  std::string model_path_;
  std::string device_;

  double min_confidence_ = 0.25;
  double score_threshold_ = 0.25;
  double nms_threshold_ = 0.45;
  float kpt_threshold_ = 0.3f;

  bool debug_ = false;

  static constexpr int input_size_ = 640;
  static constexpr int class_num_ = 1;
};

}  // namespace detector
