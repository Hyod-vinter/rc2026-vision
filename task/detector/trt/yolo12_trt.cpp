#include "yolo12_trt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "postprocess.h"
#include "preprocess.h"
#include "tools/yaml.hpp"

namespace detector
{
namespace
{
Logger gLogger;

// 这里直接用 config.h 里的参数计算输出 buffer 大小
constexpr int kOutputSize =
  kMaxNumOutputBbox * static_cast<int>(sizeof(Detection) / sizeof(float)) + 1;

inline void check_trt(bool ok, const std::string & msg)
{
  if (!ok) {
    throw std::runtime_error(msg);
  }
}

inline cv::Rect clamp_rect_to_image(const cv::Rect & r, const cv::Mat & img)
{
  const cv::Rect img_rect(0, 0, img.cols, img.rows);
  return r & img_rect;
}

  std::string dims_to_string(const nvinfer1::Dims & dims)
{
  std::string s = "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    s += std::to_string(dims.d[i]);
    if (i + 1 != dims.nbDims) s += ", ";
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

void dump_io_tensors(nvinfer1::ICudaEngine * engine, nvinfer1::IExecutionContext * context)
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

yolo12_trt::yolo12_trt(const std::string & config_path, bool debug)
: debug_(debug)
{
  auto yaml = tools::load(config_path);

  model_path_ = tools::read<std::string>(yaml, "trt_model_path");
  device_ = tools::read<std::string>(yaml, "trt_device");

  // 推理阈值直接使用 config.h
  min_confidence_ = kConfThresh;

  const int x = yaml["roi"]["x"].as<int>();
  const int y = yaml["roi"]["y"].as<int>();
  const int width = yaml["roi"]["width"].as<int>();
  const int height = yaml["roi"]["height"].as<int>();

  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(static_cast<float>(x), static_cast<float>(y));

  save_path_ = "imgs";
  std::filesystem::create_directories(save_path_);

  if (device_.rfind("GPU", 0) == 0 || device_.rfind("gpu", 0) == 0) {
    CUDA_CHECK(cudaSetDevice(kGpuId));
  }

  load_engine();
}

yolo12_trt::~yolo12_trt()
{
  release();
}

void yolo12_trt::load_engine()
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

  // 如果输入是动态 shape，先设置实际输入 shape
  {
    const auto in_dims = engine_->getTensorShape(kInputTensorName);
    bool has_dynamic = false;
    for (int i = 0; i < in_dims.nbDims; ++i) {
      if (in_dims.d[i] < 0) {
        has_dynamic = true;
        break;
      }
    }

    if (has_dynamic) {
      nvinfer1::Dims4 actual_input_dims{1, 3, kInputH, kInputW};
      bool ok = context_->setInputShape(kInputTensorName, actual_input_dims);
      check_trt(ok, "setInputShape failed");
    }
  }

  dump_io_tensors(engine_, context_);

  {
  const auto out_dims = context_->getTensorShape(kOutputTensorName);

  output_dtype_ = engine_->getTensorDataType(kOutputTensorName);
  output_numel_ = dims_volume(out_dims);
  output_bytes_ = output_numel_ * dtype_size(output_dtype_);

  std::cout
    << "[TRT] output dims=" << dims_to_string(out_dims)
    << ", numel=" << output_numel_
    << ", dtype=" << dtype_to_string(output_dtype_)
    << ", bytes=" << output_bytes_
    << std::endl;

  int num_boxes = -1;
  int num_attrs = -1;

  if (out_dims.nbDims == 3 && out_dims.d[0] == 1) {
    if (out_dims.d[1] < out_dims.d[2]) {
      num_attrs = out_dims.d[1];
      num_boxes = out_dims.d[2];
      std::cout << "[TRT] layout = [1, attrs, boxes]" << std::endl;
    } else {
      num_boxes = out_dims.d[1];
      num_attrs = out_dims.d[2];
      std::cout << "[TRT] layout = [1, boxes, attrs]" << std::endl;
    }
  } else if (out_dims.nbDims == 2) {
    if (out_dims.d[0] > out_dims.d[1]) {
      num_boxes = out_dims.d[0];
      num_attrs = out_dims.d[1];
      std::cout << "[TRT] layout = [boxes, attrs]" << std::endl;
    } else {
      num_attrs = out_dims.d[0];
      num_boxes = out_dims.d[1];
      std::cout << "[TRT] layout = [attrs, boxes]" << std::endl;
    }
  } else {
    throw std::runtime_error("unsupported output dims: " + dims_to_string(out_dims));
  }

  model_bboxes_ = num_boxes;

  decode_mode_ = TrtDecodeMode::OnnxXYWH;

  std::cout << "[TRT] decode_mode=OnnxXYWH" << std::endl;

}

  CUDA_CHECK(cudaStreamCreate(&stream_));

  cuda_preprocess_init(preprocess_ctx_, kMaxInputImageSize);

  // 输入 buffer
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&input_buffer_device_),
    kBatchSize * 3 * kInputH * kInputW * sizeof(float)));

  // 原始 TRT 输出 buffer：按真实输出 tensor 分配
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&output_buffer_device_),
    output_bytes_));

  // decode + NMS 输出 buffer
  decode_ptr_host_ = new float[1 + kMaxNumOutputBbox * bbox_element];
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&decode_ptr_device_),
    sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
}

void yolo12_trt::release()
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

std::list<KFS> yolo12_trt::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    return {};
  }

  tmp_img_ = raw_img;

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {
      roi_.height = raw_img.rows;
    }

    const cv::Rect valid_roi = roi_ & cv::Rect(0, 0, raw_img.cols, raw_img.rows);
    if (valid_roi.width <= 0 || valid_roi.height <= 0) {
      return {};
    }

    // 注意：ROI 子图可能不是连续内存，给 GPU 预处理前先 clone
    bgr_img = raw_img(valid_roi).clone();
  } else {
    // 统一 clone，保证连续内存，便于 GPU 预处理
    bgr_img = raw_img.clone();
  }

  // ========== GPU 预处理 ==========
  cuda_preprocess(
    preprocess_ctx_,
    bgr_img.data,
    bgr_img.cols,
    bgr_img.rows,
    input_buffer_device_,
    kInputW,
    kInputH,
    stream_);

  // ========== TRT 推理 ==========
  bool ok = context_->setInputTensorAddress(
    kInputTensorName,
    static_cast<void *>(input_buffer_device_));
  check_trt(ok, "TensorRT setInputTensorAddress failed");

  ok = context_->setTensorAddress(
    kOutputTensorName,
    static_cast<void *>(output_buffer_device_));
  check_trt(ok, "TensorRT setTensorAddress failed");

  ok = context_->enqueueV3(stream_);
  check_trt(ok, "TensorRT enqueueV3 failed");

  // ========== GPU decode + GPU NMS ==========
  CUDA_CHECK(cudaMemsetAsync(
    decode_ptr_device_,
    0,
    sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element),
    stream_));

  if (decode_mode_ == TrtDecodeMode::OnnxXYWH) {
    cuda_decode_onnx_xywh(
      static_cast<const float*>(output_buffer_device_),
      model_bboxes_,
      kNumClass,
      kConfThresh,
      decode_ptr_device_,
      kMaxNumOutputBbox,
      stream_);
  } else if (decode_mode_ == TrtDecodeMode::RawLTRB) {
    cuda_decode_raw_head(
      static_cast<const float*>(output_buffer_device_),
      model_bboxes_,
      kInputW,
      kInputH,
      kNumClass,
      kConfThresh,
      decode_ptr_device_,
      kMaxNumOutputBbox,
      stream_);
  } else {
    cuda_decode(
      static_cast<float*>(output_buffer_device_),
      model_bboxes_,
      kConfThresh,
      decode_ptr_device_,
      kMaxNumOutputBbox,
      stream_);
  }


  cuda_nms(
    decode_ptr_device_,
    kNmsThresh,
    kMaxNumOutputBbox,
    stream_);

  CUDA_CHECK(cudaMemcpyAsync(
    decode_ptr_host_,
    decode_ptr_device_,
    sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element),
    cudaMemcpyDeviceToHost,
    stream_));
  
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return parse_trt_output(decode_ptr_host_, bgr_img, frame_count);
}

std::list<KFS> yolo12_trt::parse_trt_output(
  const float * decode_ptr_host,
  const cv::Mat & bgr_img,
  int frame_count)
{
  std::list<KFS> kfss;

  if (decode_ptr_host == nullptr) {
    return kfss;
  }

  const int count = std::min(static_cast<int>(decode_ptr_host[0]), kMaxNumOutputBbox);

  for (int i = 0; i < count; ++i) {
    const int base = 1 + i * bbox_element;
    const int keep_flag = static_cast<int>(decode_ptr_host[base + 6]);

    if (keep_flag != 1) {
      continue;
    }

    const float conf = decode_ptr_host[base + 4];
    const int class_id = static_cast<int>(decode_ptr_host[base + 5]);

    if (std::isnan(conf) || conf < static_cast<float>(min_confidence_)) {
      continue;
    }

    float bbox[4];
    bbox[0] = decode_ptr_host[base + 0];  // left  (network space)
    bbox[1] = decode_ptr_host[base + 1];  // top
    bbox[2] = decode_ptr_host[base + 2];  // right
    bbox[3] = decode_ptr_host[base + 3];  // bottom

    cv::Mat rect_img = bgr_img;
    cv::Rect local_box = get_rect(rect_img, bbox);
    local_box = clamp_rect_to_image(local_box, bgr_img);

    if (local_box.width <= 0 || local_box.height <= 0) {
      continue;
    }

    KFS kfs;
    kfs.class_id = class_id;
    kfs.confidence = conf;

    // pattern 在 ROI 局部图上截取
    kfs.pattern = bgr_img(local_box).clone();

    if (use_roi_) {
      // 还原到原图坐标
      local_box.x += static_cast<int>(offset_.x);
      local_box.y += static_cast<int>(offset_.y);
    }

    kfs.box = local_box;
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

    kfss.emplace_back(std::move(kfs));
  }

  if (debug_) {
    draw_detections(use_roi_ ? tmp_img_ : bgr_img, kfss, frame_count);
  }

  return kfss;
}

bool yolo12_trt::check_name(const KFS & kfs) const
{
  return kfs.class_id >= 0 && kfs.class_id < kNumClass;
}

bool yolo12_trt::check_confidence(const KFS & kfs) const
{
  return kfs.confidence > kConfThresh;
}

cv::Point2f yolo12_trt::get_center(const cv::Rect & box) const
{
  return {
    box.x + box.width * 0.5f,
    box.y + box.height * 0.5f
  };
}

cv::Point2f yolo12_trt::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  const auto h = static_cast<float>(bgr_img.rows);
  const auto w = static_cast<float>(bgr_img.cols);
  return {center.x / w, center.y / h};
}

KFSname yolo12_trt::get_name(int class_id) const
{
  if (class_id < 0 || class_id >= kNumClass) {
    return KFSname::UNKNOWN;
  }
  return static_cast<KFSname>(class_id);
}

Colors yolo12_trt::get_color(const cv::Mat & pattern) const
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

void yolo12_trt::draw_detections(
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

void yolo12_trt::save(const KFS & kfs) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format(
    "{}/{}_{}.jpg",
    save_path_,
    KFS_NAME[static_cast<int>(kfs.name)],
    file_name);

  cv::imwrite(img_path, tmp_img_);
}

}  // namespace detector
