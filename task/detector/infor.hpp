#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace detector
{

// classes.txt:
// pingmu, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, C, E
enum Inforname
{
  PINGMU,
  NUM_0,
  NUM_1,
  NUM_2,
  NUM_3,
  NUM_4,
  NUM_5,
  NUM_6,
  NUM_7,
  NUM_8,
  NUM_9,
  LETTER_A,
  LETTER_C,
  LETTER_E,
  INFOR_UNKNOWN
};

const std::vector<std::string> INFOR_NAME = {
  "pingmu",
  "0",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "A",
  "C",
  "E",
  "UNKNOWN"
};

enum InforCommandType
{
  INFOR_STAGE_COMMAND,    // 阶段指令：A0-A9, C0-C9, E0-E9
  INFOR_SPECIAL_COMMAND,  // 特殊指令：1-2970
  INFOR_COMMAND_UNKNOWN
};

const std::vector<std::string> INFOR_COMMAND_TYPE = {
  "STAGE",
  "SPECIAL",
  "UNKNOWN"
};

struct infor
{
  Inforname name{INFOR_UNKNOWN};
  cv::Point2f center;
  cv::Point2f center_norm;
  int class_id{-1};
  double confidence{0.0};
  cv::Rect box;
  cv::Mat pattern;

  // 串口屏向外扩展后的 ROI，以及 ROI 内从左到右拼接得到的指令
  cv::Rect roi;
  std::vector<infor> chars;
  std::string command;
  InforCommandType command_type{INFOR_COMMAND_UNKNOWN};
  int stage_value{-1};    // 阶段指令映射值：A0-A9=2971-2980, C0-C9=2981-2990, E0-E9=2991-3000
  int special_value{-1};  // 特殊指令数值：1-2970
  bool command_valid{false};

  static Inforname class_id_to_name(int class_id);
  static std::string name_to_text(Inforname name);
  static InforCommandType command_to_type(const std::string & command);

  bool is_screen() const;
  bool is_char() const;
  std::string text() const;
  void update_name_from_class_id();
  void clear_command();

  // 按串口屏位置向外扩展，得到 ROI 感兴趣区
  cv::Rect expanded_roi(
    const cv::Size & image_size,
    int expand_x = 20,
    int expand_y = 20) const;

  static cv::Rect make_roi_from_screen(
    const infor & screen,
    const cv::Size & image_size,
    int expand_x = 20,
    int expand_y = 20);

  static bool point_in_roi(
    const cv::Point2f & point,
    const cv::Rect & roi);

  static bool detection_in_roi(
    const infor & det,
    const cv::Rect & roi);

  // 在 detections 中选择置信度最高的 pingmu 作为串口屏
  static const infor * find_best_screen(
    const std::vector<infor> & detections,
    double min_confidence = 0.0);

  // 使用指定 screen 生成指令：筛选 ROI 内字符，按 x 坐标从左到右拼接
  static std::string build_command_string(
    const std::vector<infor> & detections,
    const infor & screen,
    const cv::Size & image_size,
    std::vector<infor> * chars_out = nullptr,
    cv::Rect * roi_out = nullptr,
    int expand_x = 20,
    int expand_y = 20,
    double min_confidence = 0.0);

  static bool is_stage_command(
    const std::string & command,
    int * value_out = nullptr);
  static int stage_command_to_value(const std::string & command);
  static bool is_special_command(
    const std::string & command,
    int * value_out = nullptr);

  static bool parse_command(
    const std::string & command,
    InforCommandType & type_out,
    int & special_value_out);

  static bool parse_command(
    const std::string & command,
    InforCommandType & type_out,
    int & special_value_out,
    int & stage_value_out);

  // 从一帧检测结果中生成完整交互信息。
  // 成功条件：找到串口屏，并拼接出的指令属于阶段指令或特殊指令。
  static bool build_from_detections(
    const std::vector<infor> & detections,
    const cv::Size & image_size,
    infor & result,
    int expand_x = 20,
    int expand_y = 20,
    double min_screen_confidence = 0.0,
    double min_char_confidence = 0.0);

  // 以当前对象作为串口屏，更新其 command/command_type/chars/roi
  bool update_command_from_detections(
    const std::vector<infor> & detections,
    const cv::Size & image_size,
    int expand_x = 20,
    int expand_y = 20,
    double min_char_confidence = 0.0);
};

}  // namespace detector