#include "wepend.hpp"

#include <fmt/format.h>
#include "tools/yaml.hpp"

namespace detector
{

Wepend::Wepend(const std::string & config_path, bool debug)
: debug_(debug)
{
  auto yaml = tools::load(config_path);

  model_path_ = tools::read<std::string>(yaml, "model_path_wepend");
  device_ = tools::read<std::string>(yaml, "device_wepend");
  min_confidence_ = tools::read<double>(yaml, "min_confidence_wepend");

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

std::list<WependObject> Wepend::detect(const cv::Mat & img, int frame_count)
{
  if (img.empty()) {
    return {};
  }

  const auto x_scale = static_cast<double>(640) / img.rows;
  const auto y_scale = static_cast<double>(640) / img.cols;
  const auto scale = std::min(x_scale, y_scale);

  const int h = static_cast<int>(img.rows * scale);
  const int w = static_cast<int>(img.cols * scale);

  cv::Mat input(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::resize(img, input(cv::Rect(0, 0, w, h)), {w, h});

  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  auto output_tensor = infer_request.get_output_tensor();
  auto shape = output_tensor.get_shape();

  cv::Mat output(shape[1], shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, img, frame_count);
}

std::list<WependObject> Wepend::parse(
  double scale, cv::Mat & output, const cv::Mat & img, int frame_count)
{
  cv::transpose(output, output);

  std::vector<int> ids;
  std::vector<float> confs;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < output.rows; ++i) {
    auto xywh = output.row(i).colRange(0, 4);
    auto scores = output.row(i).colRange(4, 4 + class_num_);

    double score;
    cv::Point cls;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &cls);

    if (score < score_threshold_) continue;

    float x = xywh.at<float>(0);
    float y = xywh.at<float>(1);
    float w = xywh.at<float>(2);
    float h = xywh.at<float>(3);

    int left = static_cast<int>((x - w * 0.5f) / scale);
    int top  = static_cast<int>((y - h * 0.5f) / scale);
    int width  = static_cast<int>(w / scale);
    int height = static_cast<int>(h / scale);

    ids.emplace_back(cls.x);
    confs.emplace_back(score);
    boxes.emplace_back(left, top, width, height);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confs, score_threshold_, nms_threshold_, indices);

  std::list<WependObject> objs;
  const auto img_center = get_image_center(img);

  for (int i : indices) {
    WependObject obj;

    obj.class_id = ids[i];
    obj.confidence = confs[i];
    obj.box = boxes[i];
    obj.name = get_name(obj.class_id);

    obj.center = get_center(obj.box);

    // 核心：偏置
    obj.offset_x = obj.center.x - img_center.x;
    obj.offset_y = obj.center.y - img_center.y;

    if (obj.confidence < min_confidence_) continue;

    objs.emplace_back(obj);
  }

  if (debug_) {
    draw_detections(img, objs, frame_count);
  }

  return objs;
}

cv::Point2f Wepend::get_center(const cv::Rect & box) const
{
  return {
    box.x + box.width * 0.5f,
    box.y + box.height * 0.5f
  };
}

cv::Point2f Wepend::get_image_center(const cv::Mat & img) const
{
  return {
    img.cols * 0.5f,
    img.rows * 0.5f
  };
}

std::string Wepend::get_name(int class_id) const
{
  switch (class_id) {
    case 0: return "weapon1";
    case 1: return "weapon2";
    case 2: return "weapon3";
    default: return "unknown";
  }
}

void Wepend::draw_detections(
  const cv::Mat & img,
  const std::list<WependObject> & objs,
  int frame_count) const
{
  cv::Mat vis = img.clone();
  auto img_center = get_image_center(img);

  cv::circle(vis, img_center, 5, cv::Scalar(255,255,0), -1);

  for (auto & o : objs) {
    cv::rectangle(vis, o.box, cv::Scalar(0,255,0), 2);
    cv::circle(vis, o.center, 5, cv::Scalar(0,0,255), -1);

    // 类别 + 置信度
    auto label = fmt::format("{} {:.2f}", o.name, o.confidence);

    // 画在框的左上角
    cv::Point text_pos(o.box.x, o.box.y - 5);
    if (text_pos.y < 0) text_pos.y = 10;

    cv::putText(vis, label, text_pos,
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);

    auto txt = fmt::format("dx:{:.1f} dy:{:.1f}", o.offset_x, o.offset_y);
    cv::putText(vis, txt, o.center,
      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
  }

  cv::imshow("wepend", vis);
}

} // namespace detector
