#include "yolo_ls.hpp"

#include <fmt/format.h>

#include <iostream>

#include "tools/yaml.hpp"

namespace detector
{

YOLOLS::YOLOLS(const std::string & config_path, bool debug)
: debug_(debug)
{
  auto yaml = tools::load(config_path);

  model_path_ = tools::read<std::string>(yaml, "model_path_ls");
  device_ = tools::read<std::string>(yaml, "device_ls");
  min_confidence_ = tools::read<double>(yaml, "min_confidence_ls");

  if (yaml["score_threshold"]) {
    score_threshold_ = yaml["score_threshold"].as<double>();
  }

  if (yaml["nms_threshold"]) {
    nms_threshold_ = yaml["nms_threshold"].as<double>();
  }

  if (yaml["kpt_threshold"]) {
    kpt_threshold_ = yaml["kpt_threshold"].as<float>();
  }

  auto model = core_.read_model(model_path_);

  std::cout << "yolo_ls input type: "
            << model->input().get_element_type()
            << std::endl;

  std::cout << "yolo_ls output type: "
            << model->output().get_element_type()
            << std::endl;

  ov::preprocess::PrePostProcessor ppp(model);

  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, input_size_, input_size_, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();

  compiled_model_ = core_.compile_model(
    model,
    device_,
    ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

std::optional<LoShuGrid> YOLOLS::detect(
  const cv::Mat & raw_img,
  int frame_count)
{
  if (raw_img.empty()) {
    return std::nullopt;
  }

  const cv::Mat & bgr_img = raw_img;

  const double x_scale = static_cast<double>(input_size_) / bgr_img.rows;
  const double y_scale = static_cast<double>(input_size_) / bgr_img.cols;
  const double scale = std::min(x_scale, y_scale);

  const int h = static_cast<int>(bgr_img.rows * scale);
  const int w = static_cast<int>(bgr_img.cols * scale);

  cv::Mat input(
    input_size_,
    input_size_,
    CV_8UC3,
    cv::Scalar(0, 0, 0));

  cv::Rect input_roi(0, 0, w, h);
  cv::resize(bgr_img, input(input_roi), cv::Size(w, h));

  ov::Tensor input_tensor(
    ov::element::u8,
    {1, input_size_, input_size_, 3},
    input.data);

  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();

  cv::Mat output(
    output_shape[1],
    output_shape[2],
    CV_32F,
    output_tensor.data());

  return parse(scale, output, bgr_img, frame_count);
}

std::optional<LoShuGrid> YOLOLS::parse(
  double scale,
  cv::Mat & output,
  const cv::Mat & bgr_img,
  int frame_count)
{
  cv::transpose(output, output);

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<std::vector<cv::Point2f>> all_kpts;
  std::vector<std::vector<float>> all_kpt_scores;

  constexpr int kpt_start = 4 + class_num_;

  for (int r = 0; r < output.rows; ++r) {
    const float score = output.at<float>(r, 4);

    if (score < score_threshold_) {
      continue;
    }

    const float x = output.at<float>(r, 0);
    const float y = output.at<float>(r, 1);
    const float w = output.at<float>(r, 2);
    const float h = output.at<float>(r, 3);

    cv::Rect box(
      static_cast<int>((x - 0.5f * w) / scale),
      static_cast<int>((y - 0.5f * h) / scale),
      static_cast<int>(w / scale),
      static_cast<int>(h / scale));

    std::vector<cv::Point2f> kpts;
    std::vector<float> kpt_scores;

    for (int i = 0; i < LO_SHU_KPT_NUM; ++i) {
      const int base = kpt_start + i * 3;

      const float kx =
        output.at<float>(r, base + 0) / static_cast<float>(scale);

      const float ky =
        output.at<float>(r, base + 1) / static_cast<float>(scale);

      const float ks =
        output.at<float>(r, base + 2);

      kpts.emplace_back(kx, ky);
      kpt_scores.emplace_back(ks);
    }

    boxes.emplace_back(box);
    confidences.emplace_back(score);
    all_kpts.emplace_back(kpts);
    all_kpt_scores.emplace_back(kpt_scores);
  }

  std::vector<int> indices;

  cv::dnn::NMSBoxes(
    boxes,
    confidences,
    score_threshold_,
    nms_threshold_,
    indices);

  if (indices.empty()) {
    return std::nullopt;
  }

  int best_index = indices[0];

  for (const int idx : indices) {
    if (confidences[idx] > confidences[best_index]) {
      best_index = idx;
    }
  }

  LoShuGrid grid;

  const bool ok = grid.update(
    boxes[best_index],
    confidences[best_index],
    0,
    all_kpts[best_index],
    all_kpt_scores[best_index],
    kpt_threshold_);

  if (!ok) {
    return std::nullopt;
  }

  if (debug_) {
    draw(bgr_img, grid, frame_count);
  }

  return grid;
}

void YOLOLS::draw(
  const cv::Mat & img,
  const LoShuGrid & grid,
  int frame_count) const
{
  cv::Mat detection = img.clone();

  if (frame_count >= 0) {
    cv::putText(
      detection,
      fmt::format("[{}]", frame_count),
      cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX,
      1.0,
      cv::Scalar(255, 255, 255),
      2);
  }

  grid.draw(
    detection,
    true,
    true,
    true,
    nullptr);

  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("yolo_ls_detection", detection);
}

}  // namespace detector
