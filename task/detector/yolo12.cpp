#include "yolo12.hpp"

#include <fmt/chrono.h>
#include <fmt/format.h>
#include "tools/yaml.hpp"

#include <filesystem>

namespace detector
{
YOLO12::YOLO12(const std::string & config_path, bool debug)
: debug_(debug)
{
  auto yaml = tools::load(config_path); 

  model_path_ = tools::read<std::string>(yaml, "model_path"); 
  device_ = tools::read<std::string>(yaml, "device");
  min_confidence_ = tools::read<double>(yaml, "min_confidence");

  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();

  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(static_cast<float>(x), static_cast<float>(y));

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  auto model = core_.read_model(model_path_);
  std::cout << "model input type: " << model->input().get_element_type() << std::endl;
  std::cout << "model output type: " << model->output().get_element_type() << std::endl;
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

std::list<KFS> YOLO12::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    return std::list<KFS>();
  }

  cv::Mat bgr_img;
  tmp_img_ = raw_img;

  if (use_roi_) {
    if (roi_.width == -1) {
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  const auto x_scale = static_cast<double>(640) / bgr_img.rows;
  const auto y_scale = static_cast<double>(640) / bgr_img.cols;
  const auto scale = std::min(x_scale, y_scale);

  const auto h = static_cast<int>(bgr_img.rows * scale);
  const auto w = static_cast<int>(bgr_img.cols * scale);

  // preprocess
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});

  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  // infer
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, bgr_img, frame_count);
}

std::list<KFS> YOLO12::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // 与你 yolo11 的 parse 风格一致：先转置，再按 row 解析
  cv::transpose(output, output);

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);

    double score = 0.0;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < score_threshold_) {
      continue;
    }

    const auto x = xywh.at<float>(0);
    const auto y = xywh.at<float>(1);
    const auto w = xywh.at<float>(2);
    const auto h = xywh.at<float>(3);

    const auto left = static_cast<int>((x - 0.5f * w) / scale);
    const auto top = static_cast<int>((y - 0.5f * h) / scale);
    const auto width = static_cast<int>(w / scale);
    const auto height = static_cast<int>(h / scale);

    ids.emplace_back(max_point.x);
    confidences.emplace_back(static_cast<float>(score));
    boxes.emplace_back(left, top, width, height);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<KFS> kfss;
  for (const auto & i : indices) {
    KFS kfs;
    kfs.class_id = ids[i];
    kfs.confidence = confidences[i];
   cv::Rect local_box = boxes[i];
    cv::Rect img_rect(0, 0, bgr_img.cols, bgr_img.rows);
    cv::Rect valid_box = local_box & img_rect;

    if (valid_box.width > 0 && valid_box.height > 0) {
    kfs.pattern = bgr_img(valid_box).clone();
    } else {
    kfs.pattern = cv::Mat();
    }
    kfs.box = local_box;

    if (use_roi_) {
    kfs.box.x += static_cast<int>(offset_.x);
    kfs.box.y += static_cast<int>(offset_.y);
    }

    kfs.center = get_center(kfs.box);
    kfs.center_norm = get_center_norm(use_roi_ ? tmp_img_ : bgr_img, kfs.center);
    kfs.name = get_name(kfs.class_id);
    kfs.workname = KFS::name_to_workname(kfs.name);
    kfs.color = get_color(kfs.pattern);

    if (!check_name(kfs)) {
      continue;
    }

    if (!check_confidence(kfs)) {
      continue;
    }

    kfss.emplace_back(kfs);
  }

  if (debug_) {
    draw_detections(use_roi_ ? tmp_img_ : bgr_img, kfss, frame_count);
  }

  return kfss;
}

bool YOLO12::check_name(const KFS & kfs) const
{
  return kfs.class_id >= 0 && kfs.class_id < class_num_;
}

bool YOLO12::check_confidence(const KFS & kfs) const
{
  return kfs.confidence > min_confidence_;
}

cv::Point2f YOLO12::get_center(const cv::Rect & box) const
{
  return {
    box.x + box.width * 0.5f,
    box.y + box.height * 0.5f
  };
}

cv::Point2f YOLO12::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  const auto h = static_cast<float>(bgr_img.rows);
  const auto w = static_cast<float>(bgr_img.cols);
  return {center.x / w, center.y / h};
}

KFSname YOLO12::get_name(int class_id) const
{
  if (class_id < 0 || class_id >= class_num_) {
    return KFSname::UNKNOWN;
  }
  return static_cast<KFSname>(class_id);
}

Colors YOLO12::get_color(const cv::Mat & pattern) const
{
  if (pattern.empty()) {
    return red;
  }

  int red_count = 0;
  int blue_count = 0;

  for (int y = 0; y < pattern.rows; ++y) {
    for (int x = 0; x < pattern.cols; ++x) {
      const cv::Vec3b & pixel = pattern.at<cv::Vec3b>(y, x);

      const int b = pixel[0];
      const int g = pixel[1];
      const int r = pixel[2];

      // 太暗的像素直接跳过
      if (b + g + r < 60) {
        continue;
      }

      if (r > b + 20 && r > g + 20) {
        ++red_count;
      } else if (b > r + 20 && b > g + 20) {
        ++blue_count;
      }
    }
  }

  return (blue_count > red_count) ? blue : red;
}

void YOLO12::draw_detections(
  const cv::Mat & img, const std::list<KFS> & kfss, int frame_count) const
{
  auto detection = img.clone();

  cv::putText(
    detection, fmt::format("[{}]", frame_count), cv::Point(10, 30),
    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

  for (const auto & kfs : kfss) {
    cv::rectangle(detection, kfs.box, cv::Scalar(0, 255, 0), 2);

    auto info = fmt::format(
      "{:.2f} {} {} {}",
      kfs.confidence,
      COLORS[static_cast<int>(kfs.color)],
      KFS_NAME[static_cast<int>(kfs.name)],
      KFS_WORKNAME[static_cast<int>(kfs.workname)]);

    cv::circle(detection, kfs.center, 5, cv::Scalar(0, 0, 255), -1);
    cv::putText(
      detection, info, kfs.center, cv::FONT_HERSHEY_SIMPLEX, 1.0,
      cv::Scalar(0, 255, 0), 2);
  }

  if (use_roi_) {
    cv::rectangle(detection, roi_, cv::Scalar(0, 255, 0), 2);
  }

  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("yolo12_kfs_detection", detection);
}

void YOLO12::save(const KFS & kfs) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format(
    "{}/{}_{}.jpg",
    save_path_,
    KFS_NAME[static_cast<int>(kfs.name)],
    file_name);

  cv::imwrite(img_path, tmp_img_);
}

std::list<KFS> YOLO12::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace detector
