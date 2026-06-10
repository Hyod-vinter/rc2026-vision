#include "yolo11_lstrt.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "cuda_utils.h"
#include "logging.h"
#include "postprocess_loshu.h"
#include "preprocess.h"
#include "tools/yaml.hpp"

namespace detector
{
namespace
{
Logger gLogger;


inline void check_trt(bool ok, const std::string & msg)
{
  if (!ok) {
    throw std::runtime_error(msg);
  }
}

std::string dims_to_string(const nvinfer1::Dims & dims)
{
  std::string s = "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    s += std::to_string(dims.d[i]);
    if (i + 1 != dims.nbDims) {
      s += ", ";
    }
  }
  s += "]";
  return s;
}

size_t dims_volume(const nvinfer1::Dims & dims)
{
  size_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("dynamic dim not set: " + dims_to_string(dims));
    }
    v *= static_cast<size_t>(dims.d[i]);
  }
  return v;
}

size_t dtype_size(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL: return 1;
    default: throw std::runtime_error("unsupported TensorRT dtype");
  }
}

const char * dtype_to_string(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return "kFLOAT";
    case nvinfer1::DataType::kHALF: return "kHALF";
    case nvinfer1::DataType::kINT8: return "kINT8";
    case nvinfer1::DataType::kINT32: return "kINT32";
    case nvinfer1::DataType::kBOOL: return "kBOOL";
    default: return "UNKNOWN";
  }
}

cv::Rect get_loshu_rect_from_net(const cv::Mat & img, const float bbox[4])
{
  const float r_w = static_cast<float>(kLoShuInputW) / static_cast<float>(img.cols);
  const float r_h = static_cast<float>(kLoShuInputH) / static_cast<float>(img.rows);

  float l = 0.0f;
  float t = 0.0f;
  float r = 0.0f;
  float b = 0.0f;

  if (r_h > r_w) {
    const float pad_y = (kLoShuInputH - r_w * img.rows) * 0.5f;
    l = bbox[0] / r_w;
    r = bbox[2] / r_w;
    t = (bbox[1] - pad_y) / r_w;
    b = (bbox[3] - pad_y) / r_w;
  } else {
    const float pad_x = (kLoShuInputW - r_h * img.cols) * 0.5f;
    l = (bbox[0] - pad_x) / r_h;
    r = (bbox[2] - pad_x) / r_h;
    t = bbox[1] / r_h;
    b = bbox[3] / r_h;
  }

  const int left = static_cast<int>(std::round(l));
  const int top = static_cast<int>(std::round(t));
  const int width = static_cast<int>(std::round(r - l));
  const int height = static_cast<int>(std::round(b - t));

  return cv::Rect(left, top, width, height);
}

cv::Point2f map_loshu_point_from_net(const cv::Mat & img, float x, float y)
{
  const float r_w = static_cast<float>(kLoShuInputW) / static_cast<float>(img.cols);
  const float r_h = static_cast<float>(kLoShuInputH) / static_cast<float>(img.rows);

  if (r_h > r_w) {
    const float pad_y = (kLoShuInputH - r_w * img.rows) * 0.5f;
    return cv::Point2f(x / r_w, (y - pad_y) / r_w);
  }

  const float pad_x = (kLoShuInputW - r_h * img.cols) * 0.5f;
  return cv::Point2f((x - pad_x) / r_h, y / r_h);
}

void dump_io_tensors(
  nvinfer1::ICudaEngine * engine,
  nvinfer1::IExecutionContext * context)
{
  const int n = engine->getNbIOTensors();

  for (int i = 0; i < n; ++i) {
    const char * name = engine->getIOTensorName(i);
    const auto mode = engine->getTensorIOMode(name);
    const auto engine_dims = engine->getTensorShape(name);
    const auto context_dims = context->getTensorShape(name);
    const auto dtype = engine->getTensorDataType(name);

    std::cout
      << (mode == nvinfer1::TensorIOMode::kINPUT ? "[INPUT]  " : "[OUTPUT] ")
      << name
      << " engine_dims=" << dims_to_string(engine_dims)
      << " context_dims=" << dims_to_string(context_dims)
      << " dtype=" << dtype_to_string(dtype)
      << std::endl;
  }
}

}  // namespace

yolo11_lstrt::yolo11_lstrt(
  const std::string & config_path,
  bool debug)
: debug_(debug)
{
  auto yaml = tools::load(config_path);

  if (yaml["trt_model_path_ls"]) {
    model_path_ = yaml["trt_model_path_ls"].as<std::string>();
  } else if (yaml["model_path_ls"]) {
    model_path_ = yaml["model_path_ls"].as<std::string>();
  } else {
    throw std::runtime_error("missing trt_model_path_ls or model_path_ls in config");
  }

  if (yaml["trt_device_ls"]) {
    device_ = yaml["trt_device_ls"].as<std::string>();
  } else if (yaml["device_ls"]) {
    device_ = yaml["device_ls"].as<std::string>();
  } else {
    device_ = "GPU";
  }

  if (yaml["score_threshold_ls"]) {
    score_threshold_ = yaml["score_threshold_ls"].as<float>();
  } else if (yaml["score_threshold"]) {
    score_threshold_ = yaml["score_threshold"].as<float>();
  }

  if (yaml["nms_threshold_ls"]) {
    nms_threshold_ = yaml["nms_threshold_ls"].as<float>();
  } else if (yaml["nms_threshold"]) {
    nms_threshold_ = yaml["nms_threshold"].as<float>();
  }

  if (yaml["kpt_threshold"]) {
    kpt_threshold_ = yaml["kpt_threshold"].as<float>();
  }

  if (device_.rfind("GPU", 0) == 0 || device_.rfind("gpu", 0) == 0) {
    CUDA_CHECK(cudaSetDevice(kLoShuGpuId));
  }

  load_engine();
}

yolo11_lstrt::~yolo11_lstrt()
{
  release();
}

void yolo11_lstrt::load_engine()
{
  std::ifstream file(model_path_, std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("failed to open TensorRT engine: " + model_path_);
  }

  file.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<char> serialized_engine(size);
  file.read(serialized_engine.data(), static_cast<std::streamsize>(size));
  file.close();

  runtime_ = nvinfer1::createInferRuntime(gLogger);
  check_trt(runtime_ != nullptr, "createInferRuntime failed");

  engine_ = runtime_->deserializeCudaEngine(serialized_engine.data(), size);
  check_trt(engine_ != nullptr, "deserializeCudaEngine failed");

  context_ = engine_->createExecutionContext();
  check_trt(context_ != nullptr, "createExecutionContext failed");

  const auto in_dims = engine_->getTensorShape(kLoShuInputTensorName);
  bool has_dynamic = false;
  for (int i = 0; i < in_dims.nbDims; ++i) {
    if (in_dims.d[i] < 0) {
      has_dynamic = true;
      break;
    }
  }

  if (has_dynamic) {
    nvinfer1::Dims4 actual_input_dims{1, 3, kLoShuInputH, kLoShuInputW};
    check_trt(
      context_->setInputShape(kLoShuInputTensorName, actual_input_dims),
      "setInputShape failed");
  }

  dump_io_tensors(engine_, context_);

  const auto out_dims = context_->getTensorShape(kLoShuOutputTensorName);

  output_dtype_ = engine_->getTensorDataType(kLoShuOutputTensorName);
  output_numel_ = dims_volume(out_dims);
  output_bytes_ = output_numel_ * dtype_size(output_dtype_);

  std::cout
    << "[yolo11_lstrt] output dims=" << dims_to_string(out_dims)
    << ", numel=" << output_numel_
    << ", dtype=" << dtype_to_string(output_dtype_)
    << ", bytes=" << output_bytes_
    << std::endl;

  int num_attrs = -1;
  int num_boxes = -1;

  if (out_dims.nbDims == 3 && out_dims.d[0] == 1) {
    if (out_dims.d[1] == kLoShuAttrNum) {
      num_attrs = out_dims.d[1];
      num_boxes = out_dims.d[2];
      std::cout << "[yolo11_lstrt] layout = [1, attrs, boxes]" << std::endl;
    } else if (out_dims.d[2] == kLoShuAttrNum) {
      num_boxes = out_dims.d[1];
      num_attrs = out_dims.d[2];
      throw std::runtime_error(
        "current cuda_decode_loshu_onnx_xywh expects [1, 35, boxes], "
        "but engine output is [1, boxes, 35]");
    }
  }

  if (num_attrs != kLoShuAttrNum || num_boxes <= 0) {
    throw std::runtime_error(
      "unexpected yolo11_lstrt output dims, expected [1, 35, 8400], got " +
      dims_to_string(out_dims));
  }

  model_bboxes_ = num_boxes;

  CUDA_CHECK(cudaStreamCreate(&stream_));
  cuda_preprocess_init(preprocess_ctx_, kLoShuMaxInputImageSize);

  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&input_buffer_device_),
    kLoShuBatchSize * 3 * kLoShuInputH * kLoShuInputW * sizeof(float)));

  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&output_buffer_device_),
    output_bytes_));

  decode_ptr_host_ = new float[kLoShuOutputSize];

  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&decode_ptr_device_),
    sizeof(float) * kLoShuOutputSize));
}

void yolo11_lstrt::release()
{
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

  if (input_buffer_device_ != nullptr) {
    cudaFree(input_buffer_device_);
    input_buffer_device_ = nullptr;
  }

  if (output_buffer_device_ != nullptr) {
    cudaFree(output_buffer_device_);
    output_buffer_device_ = nullptr;
  }

  if (decode_ptr_device_ != nullptr) {
    cudaFree(decode_ptr_device_);
    decode_ptr_device_ = nullptr;
  }

  if (decode_ptr_host_ != nullptr) {
    delete[] decode_ptr_host_;
    decode_ptr_host_ = nullptr;
  }

  cuda_preprocess_destroy(preprocess_ctx_);

  if (context_ != nullptr) {
    delete context_;
    context_ = nullptr;
  }
  if (engine_ != nullptr) {
    delete engine_;
    engine_ = nullptr;
  }
  if (runtime_ != nullptr) {
    delete runtime_;
    runtime_ = nullptr;
  }
}

std::optional<LoShuGrid> yolo11_lstrt::detect(
  const cv::Mat & raw_img,
  int frame_count)
{
  if (raw_img.empty()) {
    return std::nullopt;
  }

  cv::Mat bgr_img = raw_img.clone();

  cuda_preprocess(
    preprocess_ctx_,
    bgr_img.data,
    bgr_img.cols,
    bgr_img.rows,
    input_buffer_device_,
    kLoShuInputW,
    kLoShuInputH,
    stream_);

  bool ok = context_->setInputTensorAddress(
    kLoShuInputTensorName,
    static_cast<void *>(input_buffer_device_));
  check_trt(ok, "TensorRT setInputTensorAddress failed");

  ok = context_->setTensorAddress(
    kLoShuOutputTensorName,
    static_cast<void *>(output_buffer_device_));
  check_trt(ok, "TensorRT setTensorAddress failed");

  ok = context_->enqueueV3(stream_);
  check_trt(ok, "TensorRT enqueueV3 failed");

  CUDA_CHECK(cudaMemsetAsync(
    decode_ptr_device_,
    0,
    sizeof(float) * kLoShuOutputSize,
    stream_));

  cuda_decode_loshu_onnx_xywh(
    static_cast<const float *>(output_buffer_device_),
    model_bboxes_,
    score_threshold_,
    decode_ptr_device_,
    kLoShuMaxNumOutputBbox,
    stream_);

  cuda_nms_loshu(
    decode_ptr_device_,
    nms_threshold_,
    kLoShuMaxNumOutputBbox,
    stream_);

  CUDA_CHECK(cudaMemcpyAsync(
    decode_ptr_host_,
    decode_ptr_device_,
    sizeof(float) * kLoShuOutputSize,
    cudaMemcpyDeviceToHost,
    stream_));

  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return parse_trt_output(decode_ptr_host_, bgr_img, frame_count);
}

std::optional<LoShuGrid> yolo11_lstrt::parse_trt_output(
  const float * decode_ptr_host,
  const cv::Mat & bgr_img,
  int frame_count)
{
  if (decode_ptr_host == nullptr) {
    return std::nullopt;
  }

  const int count = std::min(
    static_cast<int>(decode_ptr_host[0]),
    kLoShuMaxNumOutputBbox);

  if (count <= 0) {
    return std::nullopt;
  }

  const int kpt_offset = 7;

  int best_base = -1;
  float best_conf = -1.0f;

  for (int i = 0; i < count; ++i) {
    const int base = 1 + i * kLoShuElement;
    const int keep = static_cast<int>(decode_ptr_host[base + 6]);
    const float conf = decode_ptr_host[base + 4];

    if (keep != 1 || std::isnan(conf) || conf < score_threshold_) {
      continue;
    }

    if (conf > best_conf) {
      best_conf = conf;
      best_base = base;
    }
  }

  if (best_base < 0) {
    return std::nullopt;
  }

  float bbox[4];
  bbox[0] = decode_ptr_host[best_base + 0];
  bbox[1] = decode_ptr_host[best_base + 1];
  bbox[2] = decode_ptr_host[best_base + 2];
  bbox[3] = decode_ptr_host[best_base + 3];

  cv::Rect box = get_loshu_rect_from_net(bgr_img, bbox);
  box = clamp_rect_to_image(box, bgr_img);

  if (box.width <= 0 || box.height <= 0) {
    return std::nullopt;
  }

  std::vector<cv::Point2f> kpts;
  std::vector<float> kpt_scores;

  kpts.reserve(kLoShuKptNum);
  kpt_scores.reserve(kLoShuKptNum);

  for (int k = 0; k < kLoShuKptNum; ++k) {
    const int p = best_base + kpt_offset + k * 3;

    const auto pt = map_loshu_point_from_net(
      bgr_img,
      decode_ptr_host[p + 0],
      decode_ptr_host[p + 1]);
    const float s = decode_ptr_host[p + 2];

    kpts.emplace_back(pt);
    kpt_scores.emplace_back(s);
  }

  LoShuGrid grid;

  const bool ok = grid.update(
    box,
    best_conf,
    0,
    kpts,
    kpt_scores,
    kpt_threshold_);

  if (!ok) {
    return std::nullopt;
  }

  if (debug_) {
    draw(bgr_img, grid, frame_count);
  }

  return grid;
}

cv::Rect yolo11_lstrt::clamp_rect_to_image(
  const cv::Rect & r,
  const cv::Mat & img)
{
  const cv::Rect img_rect(0, 0, img.cols, img.rows);
  return r & img_rect;
}

void yolo11_lstrt::draw(
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

  grid.draw(detection, true, true, true, nullptr);

  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("yolo11_lstrt_detection", detection);
}

}  // namespace detector
