#include "yolo12_numtrt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include "config_num.hpp"
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

cv::Rect get_num_rect(const cv::Mat & img, const float bbox[4])
{
  const float r_w = static_cast<float>(knumInputW) / static_cast<float>(img.cols);
  const float r_h = static_cast<float>(knumInputH) / static_cast<float>(img.rows);

  float left = 0.0f;
  float right = 0.0f;
  float top = 0.0f;
  float bottom = 0.0f;

  if (r_h > r_w) {
    const float pad_h =
      (static_cast<float>(knumInputH) - r_w * static_cast<float>(img.rows)) * 0.5f;
    left = bbox[0] / r_w;
    right = bbox[2] / r_w;
    top = (bbox[1] - pad_h) / r_w;
    bottom = (bbox[3] - pad_h) / r_w;
  } else {
    const float pad_w =
      (static_cast<float>(knumInputW) - r_h * static_cast<float>(img.cols)) * 0.5f;
    left = (bbox[0] - pad_w) / r_h;
    right = (bbox[2] - pad_w) / r_h;
    top = bbox[1] / r_h;
    bottom = bbox[3] / r_h;
  }

  return cv::Rect(
    static_cast<int>(std::round(left)),
    static_cast<int>(std::round(top)),
    static_cast<int>(std::round(right - left)),
    static_cast<int>(std::round(bottom - top)));
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

std::string read_yaml_string_with_fallback(
  const YAML::Node & yaml,
  const std::vector<std::string> & keys)
{
  for (const auto & key : keys) {
    if (yaml[key]) {
      return yaml[key].as<std::string>();
    }
  }

  throw std::runtime_error("missing required yaml key: " + keys.front());
}

}  // namespace

yolo12num_trt::yolo12num_trt(const std::string & config_path, bool debug)
: debug_(debug)
{
  auto yaml = tools::load(config_path);

  model_path_ = tools::read<std::string>(yaml, "trt_model_path_num");
  device_ = tools::read<std::string>(yaml, "trt_device_num");

  min_confidence_ = knumConfThresh;

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
    CUDA_CHECK(cudaSetDevice(knumGpuId));
  }

  load_engine();
}

yolo12num_trt::~yolo12num_trt()
{
  release();
}

void yolo12num_trt::load_engine()
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

  // 如果输入是动态 shape，先设置实际输入 shape。
  {
    const auto in_dims = engine_->getTensorShape(knumInputTensorName);
    bool has_dynamic = false;
    for (int i = 0; i < in_dims.nbDims; ++i) {
      if (in_dims.d[i] < 0) {
        has_dynamic = true;
        break;
      }
    }

    if (has_dynamic) {
      nvinfer1::Dims4 actual_input_dims{1, 3, knumInputH, knumInputW};
      const bool ok = context_->setInputShape(knumInputTensorName, actual_input_dims);
      check_trt(ok, "setInputShape failed");
    }
  }

  dump_io_tensors(engine_, context_);

  {
    const auto out_dims = context_->getTensorShape(knumOutputTensorName);

    output_dtype_ = engine_->getTensorDataType(knumOutputTensorName);
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
    decode_mode_ = NumTrtDecodeMode::OnnxXYWH;

    std::cout
      << "[TRT] bboxes=" << num_boxes
      << ", attrs=" << num_attrs
      << ", decode_mode=OnnxXYWH"
      << std::endl;
  }

  CUDA_CHECK(cudaStreamCreate(&stream_));

  cuda_preprocess_init(preprocess_ctx_, knumMaxInputImageSize);

  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&input_buffer_device_),
    knumBatchSize * 3 * knumInputH * knumInputW * sizeof(float)));

  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&output_buffer_device_),
    output_bytes_));

  decode_ptr_host_ = new float[1 + knumMaxNumOutputBbox * bbox_element];
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&decode_ptr_device_),
    sizeof(float) * (1 + knumMaxNumOutputBbox * bbox_element)));
}

void yolo12num_trt::release()
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

std::list<num> yolo12num_trt::detect(const cv::Mat & raw_img, int frame_count)
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

    // ROI 子图可能不是连续内存，给 GPU 预处理前先 clone。
    bgr_img = raw_img(valid_roi).clone();
  } else {
    // 统一 clone，保证连续内存，便于 GPU 预处理。
    bgr_img = raw_img.clone();
  }

  // ========== GPU 预处理 ==========
  cuda_preprocess(
    preprocess_ctx_,
    bgr_img.data,
    bgr_img.cols,
    bgr_img.rows,
    input_buffer_device_,
    knumInputW,
    knumInputH,
    stream_);

  // ========== TRT 推理 ==========
  bool ok = context_->setInputTensorAddress(
    knumInputTensorName,
    static_cast<void *>(input_buffer_device_));
  check_trt(ok, "TensorRT setInputTensorAddress failed");

  ok = context_->setTensorAddress(
    knumOutputTensorName,
    static_cast<void *>(output_buffer_device_));
  check_trt(ok, "TensorRT setTensorAddress failed");

  ok = context_->enqueueV3(stream_);
  check_trt(ok, "TensorRT enqueueV3 failed");

  // ========== GPU decode + GPU NMS ==========
  CUDA_CHECK(cudaMemsetAsync(
    decode_ptr_device_,
    0,
    sizeof(float) * (1 + knumMaxNumOutputBbox * bbox_element),
    stream_));

  if (decode_mode_ == NumTrtDecodeMode::OnnxXYWH) {
    cuda_decode_onnx_xywh(
      static_cast<const float *>(output_buffer_device_),
      model_bboxes_,
      knumNumClass,
      knumConfThresh,
      decode_ptr_device_,
      knumMaxNumOutputBbox,
      stream_);
  } else if (decode_mode_ == NumTrtDecodeMode::RawLTRB) {
    cuda_decode_raw_head(
      static_cast<const float *>(output_buffer_device_),
      model_bboxes_,
      knumInputW,
      knumInputH,
      knumNumClass,
      knumConfThresh,
      decode_ptr_device_,
      knumMaxNumOutputBbox,
      stream_);
  } else {
    cuda_decode(
      static_cast<float *>(output_buffer_device_),
      model_bboxes_,
      knumConfThresh,
      decode_ptr_device_,
      knumMaxNumOutputBbox,
      stream_);
  }

  cuda_nms(
    decode_ptr_device_,
    knumNmsThresh,
    knumMaxNumOutputBbox,
    stream_);

  CUDA_CHECK(cudaMemcpyAsync(
    decode_ptr_host_,
    decode_ptr_device_,
    sizeof(float) * (1 + knumMaxNumOutputBbox * bbox_element),
    cudaMemcpyDeviceToHost,
    stream_));

  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return parse_trt_output(decode_ptr_host_, bgr_img, frame_count);
}

std::list<num> yolo12num_trt::parse_trt_output(
  const float * decode_ptr_host,
  const cv::Mat & bgr_img,
  int frame_count)
{
  std::list<num> nums;

  if (decode_ptr_host == nullptr) {
    return nums;
  }

  const int count = std::min(static_cast<int>(decode_ptr_host[0]), knumMaxNumOutputBbox);

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

    cv::Rect local_box = get_num_rect(bgr_img, bbox);
    local_box = clamp_rect_to_image(local_box, bgr_img);

    if (local_box.width <= 0 || local_box.height <= 0) {
      continue;
    }

    num num_obj;
    num_obj.class_id = class_id;
    num_obj.confidence = conf;
    num_obj.name = get_name(class_id);
    num_obj.name_str = get_name_string(num_obj.name);

    if (use_roi_) {
      // 还原到原图坐标。
      local_box.x += static_cast<int>(offset_.x);
      local_box.y += static_cast<int>(offset_.y);
    }

    num_obj.box = local_box;
    num_obj.center = get_center(num_obj.box);
    num_obj.offset_y =
      num_obj.center.y - static_cast<float>((use_roi_ ? tmp_img_.rows : bgr_img.rows)) * 0.5f;

    if (!check_name(num_obj)) {
      continue;
    }
    if (!check_confidence(num_obj)) {
      continue;
    }

    nums.emplace_back(std::move(num_obj));
  }

  if (debug_) {
    draw_detections(use_roi_ ? tmp_img_ : bgr_img, nums, frame_count);
  }

  return nums;
}

bool yolo12num_trt::check_name(const num & num_obj) const
{
  return num_obj.class_id >= 0 &&
    num_obj.class_id < knumNumClass &&
    num_obj.name != NUM_UNKNOWN;
}

bool yolo12num_trt::check_confidence(const num & num_obj) const
{
  return num_obj.confidence > knumConfThresh;
}

cv::Point2f yolo12num_trt::get_center(const cv::Rect & box) const
{
  return {
    box.x + box.width * 0.5f,
    box.y + box.height * 0.5f
  };
}

numname yolo12num_trt::get_name(int class_id) const
{
  if (class_id < 0 ||
    class_id >= knumNumClass ||
    class_id >= static_cast<int>(NUM_NAME.size()) - 1)
  {
    return NUM_UNKNOWN;
  }

  return static_cast<numname>(class_id);
}

std::string yolo12num_trt::get_name_string(int class_id) const
{
  return get_name_string(get_name(class_id));
}

std::string yolo12num_trt::get_name_string(numname name) const
{
  const int idx = static_cast<int>(name);
  if (idx < 0 || idx >= static_cast<int>(NUM_NAME.size())) {
    return NUM_NAME.back();
  }
  return NUM_NAME[static_cast<size_t>(idx)];
}

void yolo12num_trt::draw_detections(
  const cv::Mat & img,
  const std::list<num> & nums,
  int frame_count) const
{
  auto detection = img.clone();

  cv::putText(
    detection, fmt::format("[{}]", frame_count), cv::Point(10, 30),
    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

  for (const auto & num_obj : nums) {
    cv::rectangle(detection, num_obj.box, cv::Scalar(0, 255, 0), 2);

    const auto text = fmt::format(
      "{:.2f} {} offset_y={:.1f}",
      num_obj.confidence,
      num_obj.name_str.empty() ? get_name_string(num_obj.name) : num_obj.name_str,
      num_obj.offset_y);

    cv::circle(detection, num_obj.center, 5, cv::Scalar(0, 0, 255), -1);
    cv::putText(
      detection, text, num_obj.center, cv::FONT_HERSHEY_SIMPLEX, 1.0,
      cv::Scalar(0, 255, 0), 2);
  }

  if (use_roi_) {
    cv::rectangle(detection, roi_, cv::Scalar(0, 255, 0), 2);
  }

  //cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("yolo12_num_detection", detection);
}

void yolo12num_trt::save(const num & num_obj) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  const auto label =
    num_obj.name_str.empty() ? get_name_string(num_obj.name) : num_obj.name_str;
  auto img_path = fmt::format(
    "{}/{}_{}.jpg",
    save_path_,
    label,
    file_name);

  cv::imwrite(img_path, tmp_img_);
}

}  // namespace detector