#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace detector
{

enum lightname
{
  red_light,
  green_light,
  blue_light,
  yellow_light,
  purple_light,
  cyan_light,
  unknown_light

};

const std::vector<std::string> LIGHT_NAME = {"red_light", "green_light", "blue_light", "yellow_light", "purple_light", "cyan_light", "unknown_light"};

struct light
{
  lightname name;
  int class_id;
  double confidence;
  cv::Rect box;
};

}  // namespace detector
