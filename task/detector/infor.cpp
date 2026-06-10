#include "infor.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace detector
{
namespace
{

bool is_stage_prefix(char c)
{
  return c == 'A' || c == 'C' || c == 'E';
}

}  // namespace

Inforname infor::class_id_to_name(int class_id)
{
  switch (class_id) {
    case 0:
      return PINGMU;
    case 1:
      return NUM_0;
    case 2:
      return NUM_1;
    case 3:
      return NUM_2;
    case 4:
      return NUM_3;
    case 5:
      return NUM_4;
    case 6:
      return NUM_5;
    case 7:
      return NUM_6;
    case 8:
      return NUM_7;
    case 9:
      return NUM_8;
    case 10:
      return NUM_9;
    case 11:
      return LETTER_A;
    case 12:
      return LETTER_C;
    case 13:
      return LETTER_E;
    default:
      return INFOR_UNKNOWN;
  }
}

std::string infor::name_to_text(Inforname name)
{
  switch (name) {
    case PINGMU:
      return "pingmu";
    case NUM_0:
      return "0";
    case NUM_1:
      return "1";
    case NUM_2:
      return "2";
    case NUM_3:
      return "3";
    case NUM_4:
      return "4";
    case NUM_5:
      return "5";
    case NUM_6:
      return "6";
    case NUM_7:
      return "7";
    case NUM_8:
      return "8";
    case NUM_9:
      return "9";
    case LETTER_A:
      return "A";
    case LETTER_C:
      return "C";
    case LETTER_E:
      return "E";
    default:
      return "UNKNOWN";
  }
}

InforCommandType infor::command_to_type(const std::string & command)
{
  if (is_stage_command(command)) {
    return INFOR_STAGE_COMMAND;
  }

  if (is_special_command(command)) {
    return INFOR_SPECIAL_COMMAND;
  }

  return INFOR_COMMAND_UNKNOWN;
}

bool infor::is_screen() const
{
  return name == PINGMU || class_id_to_name(class_id) == PINGMU;
}

bool infor::is_char() const
{
  const Inforname n = (name != INFOR_UNKNOWN) ? name : class_id_to_name(class_id);
  return n == NUM_0 || n == NUM_1 || n == NUM_2 || n == NUM_3 || n == NUM_4 ||
         n == NUM_5 || n == NUM_6 || n == NUM_7 || n == NUM_8 || n == NUM_9 ||
         n == LETTER_A || n == LETTER_C || n == LETTER_E;
}

std::string infor::text() const
{
  const Inforname n = (name != INFOR_UNKNOWN) ? name : class_id_to_name(class_id);
  return name_to_text(n);
}

void infor::update_name_from_class_id()
{
  name = class_id_to_name(class_id);
}

void infor::clear_command()
{
  roi = cv::Rect();
  chars.clear();
  command.clear();
  command_type = INFOR_COMMAND_UNKNOWN;
  stage_value = -1;
  special_value = -1;
  command_valid = false;
}

cv::Rect infor::expanded_roi(
  const cv::Size & image_size,
  int expand_x,
  int expand_y) const
{
  return make_roi_from_screen(*this, image_size, expand_x, expand_y);
}

cv::Rect infor::make_roi_from_screen(
  const infor & screen,
  const cv::Size & image_size,
  int expand_x,
  int expand_y)
{
  if (image_size.width <= 0 || image_size.height <= 0 || screen.box.area() <= 0) {
    return cv::Rect();
  }

  expand_x = std::max(0, expand_x);
  expand_y = std::max(0, expand_y);

  const int x0 = std::max(0, screen.box.x - expand_x);
  const int y0 = std::max(0, screen.box.y - expand_y);
  const int x1 = std::min(image_size.width, screen.box.x + screen.box.width + expand_x);
  const int y1 = std::min(image_size.height, screen.box.y + screen.box.height + expand_y);

  if (x1 <= x0 || y1 <= y0) {
    return cv::Rect();
  }

  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

bool infor::point_in_roi(
  const cv::Point2f & point,
  const cv::Rect & roi)
{
  if (roi.area() <= 0) {
    return false;
  }

  return point.x >= static_cast<float>(roi.x) &&
         point.y >= static_cast<float>(roi.y) &&
         point.x < static_cast<float>(roi.x + roi.width) &&
         point.y < static_cast<float>(roi.y + roi.height);
}

bool infor::detection_in_roi(
  const infor & det,
  const cv::Rect & roi)
{
  if (roi.area() <= 0) {
    return false;
  }

  if (point_in_roi(det.center, roi)) {
    return true;
  }

  // 若中心点因为检测框偏移略出 ROI，但检测框与 ROI 有交集，也保留
  const cv::Rect inter = det.box & roi;
  return inter.area() > 0;
}

const infor * infor::find_best_screen(
  const std::vector<infor> & detections,
  double min_confidence)
{
  const infor * best = nullptr;
  double best_confidence = -std::numeric_limits<double>::infinity();

  for (const auto & det : detections) {
    if (!det.is_screen() || det.confidence < min_confidence) {
      continue;
    }

    if (det.confidence > best_confidence) {
      best = &det;
      best_confidence = det.confidence;
    }
  }

  return best;
}

std::string infor::build_command_string(
  const std::vector<infor> & detections,
  const infor & screen,
  const cv::Size & image_size,
  std::vector<infor> * chars_out,
  cv::Rect * roi_out,
  int expand_x,
  int expand_y,
  double min_confidence)
{
  const cv::Rect roi = make_roi_from_screen(screen, image_size, expand_x, expand_y);

  if (roi_out != nullptr) {
    *roi_out = roi;
  }

  if (chars_out != nullptr) {
    chars_out->clear();
  }

  if (roi.area() <= 0) {
    return "";
  }

  std::vector<infor> valid_chars;
  valid_chars.reserve(detections.size());

  for (const auto & det : detections) {
    if (!det.is_char() || det.confidence < min_confidence) {
      continue;
    }

    if (!detection_in_roi(det, roi)) {
      continue;
    }

    valid_chars.emplace_back(det);
  }

  std::stable_sort(
    valid_chars.begin(),
    valid_chars.end(),
    [](const infor & lhs, const infor & rhs) {
      if (std::fabs(lhs.center.x - rhs.center.x) > 1e-3f) {
        return lhs.center.x < rhs.center.x;
      }
      return lhs.center.y < rhs.center.y;
    });

  std::string command;
  command.reserve(valid_chars.size());
  for (const auto & ch : valid_chars) {
    const std::string t = ch.text();
    if (t.size() == 1) {
      command += t;
    }
  }

  if (chars_out != nullptr) {
    *chars_out = valid_chars;
  }

  return command;
}

int infor::stage_command_to_value(const std::string & command)
{
  if (command.size() != 2 || !is_stage_prefix(command[0]) ||
      command[1] < '0' || command[1] > '9') {
    return -1;
  }

  const int digit = command[1] - '0';

  switch (command[0]) {
    case 'A':
      return 2971 + digit;
    case 'C':
      return 2981 + digit;
    case 'E':
      return 2991 + digit;
    default:
      return -1;
  }
}

bool infor::is_stage_command(
  const std::string & command,
  int * value_out)
{
  const int value = stage_command_to_value(command);

  if (value_out != nullptr) {
    *value_out = value;
  }

  return value != -1;
}

bool infor::is_special_command(
  const std::string & command,
  int * value_out)
{
  if (value_out != nullptr) {
    *value_out = -1;
  }

  if (command.empty()) {
    return false;
  }

  // 特殊指令为 1 到 2970，不接收 01、001 这类非规范数字串
  if (command.size() > 1 && command[0] == '0') {
    return false;
  }

  for (const char c : command) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      return false;
    }
  }

  long value = 0;
  try {
    value = std::stol(command);
  } catch (...) {
    return false;
  }

  if (value < 1 || value > 2970) {
    return false;
  }

  if (value_out != nullptr) {
    *value_out = static_cast<int>(value);
  }

  return true;
}

bool infor::parse_command(
  const std::string & command,
  InforCommandType & type_out,
  int & special_value_out)
{
  int stage_value_unused = -1;
  return parse_command(command, type_out, special_value_out, stage_value_unused);
}

bool infor::parse_command(
  const std::string & command,
  InforCommandType & type_out,
  int & special_value_out,
  int & stage_value_out)
{
  type_out = INFOR_COMMAND_UNKNOWN;
  special_value_out = -1;
  stage_value_out = -1;

  int stage_value = -1;
  if (is_stage_command(command, &stage_value)) {
    type_out = INFOR_STAGE_COMMAND;
    stage_value_out = stage_value;
    return true;
  }

  int value = -1;
  if (is_special_command(command, &value)) {
    type_out = INFOR_SPECIAL_COMMAND;
    special_value_out = value;
    return true;
  }

  return false;
}

bool infor::build_from_detections(
  const std::vector<infor> & detections,
  const cv::Size & image_size,
  infor & result,
  int expand_x,
  int expand_y,
  double min_screen_confidence,
  double min_char_confidence)
{
  result.clear_command();

  const infor * screen = find_best_screen(detections, min_screen_confidence);
  if (screen == nullptr) {
    result.name = INFOR_UNKNOWN;
    return false;
  }

  result = *screen;
  result.clear_command();

  result.command = build_command_string(
    detections,
    *screen,
    image_size,
    &result.chars,
    &result.roi,
    expand_x,
    expand_y,
    min_char_confidence);

  result.command_valid = parse_command(
    result.command,
    result.command_type,
    result.special_value,
    result.stage_value);

  return result.command_valid;
}

bool infor::update_command_from_detections(
  const std::vector<infor> & detections,
  const cv::Size & image_size,
  int expand_x,
  int expand_y,
  double min_char_confidence)
{
  clear_command();

  if (!is_screen()) {
    return false;
  }

  command = build_command_string(
    detections,
    *this,
    image_size,
    &chars,
    &roi,
    expand_x,
    expand_y,
    min_char_confidence);

  command_valid = parse_command(
    command,
    command_type,
    special_value,
    stage_value);

  return command_valid;
}

}  // namespace detector