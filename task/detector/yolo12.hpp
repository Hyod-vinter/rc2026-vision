#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "KFS.hpp"

namespace detector
{

class YOLO12
{
public:
  YOLO12(const std::string & config_path, bool debug = false);

  std::list<KFS> detect(const cv::Mat & raw_img, int frame_count = -1);

  std::list<KFS> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

private:
  std::string device_, model_path_;
  std::string save_path_;
  bool debug_{false};
  bool use_roi_{false};

  // KFSname 类别数量
  static constexpr int class_num_ = 32;
  static constexpr float nms_threshold_ = 0.5f;
  static constexpr float score_threshold_ = 0.1f;

  double min_confidence_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  cv::Rect roi_;
  cv::Point2f offset_{0.f, 0.f};
  cv::Mat tmp_img_;

private:
  std::list<KFS> parse(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  bool check_name(const KFS & kfs) const;
  bool check_confidence(const KFS & kfs) const;

  cv::Point2f get_center(const cv::Rect & box) const;
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  KFSname get_name(int class_id) const;
  Colors get_color(const cv::Mat & pattern) const;

  void draw_detections(const cv::Mat & img, const std::list<KFS> & kfss, int frame_count) const;
  void save(const KFS & kfs) const;
};

}  // namespace detector
