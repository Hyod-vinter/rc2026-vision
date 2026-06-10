#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

namespace detector
{

struct WependObject
{
  int class_id{-1};
  float confidence{0.f};

  cv::Rect box;
  cv::Point2f center;

  float offset_x{0.f};
  float offset_y{0.f};

  std::string name;
};

class Wepend
{
public:
  Wepend(const std::string & config_path, bool debug = false);

  std::list<WependObject> detect(const cv::Mat & raw_img, int frame_count = -1);

private:
  std::string device_, model_path_;
  bool debug_{false};

  static constexpr int class_num_ = 3;
  static constexpr float nms_threshold_ = 0.5f;
  static constexpr float score_threshold_ = 0.7f;

  double min_confidence_{0.1};

  ov::Core core_;
  ov::CompiledModel compiled_model_;

private:
  std::list<WependObject> parse(
    double scale, cv::Mat & output, const cv::Mat & img, int frame_count);

  cv::Point2f get_center(const cv::Rect & box) const;
  cv::Point2f get_image_center(const cv::Mat & img) const;

  std::string get_name(int class_id) const;

  void draw_detections(
    const cv::Mat & img, const std::list<WependObject> & objs, int frame_count) const;
};

}  // namespace detector
