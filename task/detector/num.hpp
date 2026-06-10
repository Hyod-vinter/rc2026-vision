#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace detector
{

// classes.txt:
// 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
// 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
//
// 注意：class_id 为 0 时对应数字 "10"，class_id 为 29 时对应数字 "39"。
enum numname
{
  NUM_10,
  NUM_11,
  NUM_12,
  NUM_13,
  NUM_14,
  NUM_15,
  NUM_16,
  NUM_17,
  NUM_18,
  NUM_19,
  NUM_20,
  NUM_21,
  NUM_22,
  NUM_23,
  NUM_24,
  NUM_25,
  NUM_26,
  NUM_27,
  NUM_28,
  NUM_29,
  NUM_30,
  NUM_31,
  NUM_32,
  NUM_33,
  NUM_34,
  NUM_35,
  NUM_36,
  NUM_37,
  NUM_38,
  NUM_39,
  NUM_UNKNOWN
};

const std::vector<std::string> NUM_NAME = {
  "10",
  "11",
  "12",
  "13",
  "14",
  "15",
  "16",
  "17",
  "18",
  "19",
  "20",
  "21",
  "22",
  "23",
  "24",
  "25",
  "26",
  "27",
  "28",
  "29",
  "30",
  "31",
  "32",
  "33",
  "34",
  "35",
  "36",
  "37",
  "38",
  "39",
  "UNKNOWN"
};

struct num
{
  int class_id{-1};
  float confidence{0.f};

  numname name{NUM_UNKNOWN};

  cv::Rect box;
  cv::Point2f center;

  // 目标中心相对图像中心的 y 方向偏移，单位：pixel。
  float offset_y{0.f};

  std::string name_str{"UNKNOWN"};
};

}  // namespace detector