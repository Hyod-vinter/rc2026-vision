#include "wependtrt.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "config_wepend.h"
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

cv::Rect get_wepend_rect(const cv::Mat & img, const float bbox[4])
{
  const float r_w = static_cast<float>(kwependInputW) / static_cast<float>(img.cols);
  const float r_h = static_cast<float>(kwependInputH) / static_cast<float>(img.rows);

  float left = 0.0f;
  float right = 0.0f;
  float top = 0.0f;
  float bottom = 0.0f;

  if (r_h > r_w) {
    const float pad_h =
      (static_cast<float>(kwependInputH) - r_w * static_cast<float>(img.rows)) * 0.5f;
    left = bbox[0] / r_w;
    right = bbox[2] / r_w;
    top = (bbox[1] - pad_h) / r_w;
    bottom = (bbox[3] - pad_h) / r_w;
  } else {
    const float pad_w =
      (static_cast<float>(kwependInputW) - r_h * static_cast<float>(img.cols)) * 0.5f;
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

WependTrt::WependTrt(const std::string & config_path, bool debug)
: debug_(debug)
{
  auto yaml = tools::load(config_path);

  model_path_ = tools::read<std::string>(yaml, "trt_model_path_wepend");
  device_ = tools::read<std::string>(yaml, "trt_device_wepend");
  min_confidence_ = kwependConfThresh;

  if (yaml["min_confidence_wepend"]) {
    min_confidence_ = yaml["min_confidence_wepend"].as<double>();
  }

  if (device_.rfind("GPU", 0) == 0 || device_.rfind("gpu", 0) == 0) {
    CUDA_CHECK(cudaSetDevice(kwependGpuId));
  }

  load_engine();
}

WependTrt::~WependTrt()
{
  release();
}

void WependTrt::load_engine()
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

  {
    const auto in_dims = engine_->getTensorShape(kwependInputTensorName);
    bool has_dynamic = false;
    for (int i = 0; i < in_dims.nbDims; ++i) {
      if (in_dims.d[i] < 0) {
        has_dynamic = true;
        break;
      }
    }

    if (has_dynamic) {
      nvinfer1::Dims4 actual_input_dims{
        kwependBatchSize,
        3,
        kwependInputH,
        kwependInputW};
      const bool ok = context_->setInputShape(kwependInputTensorName, actual_input_dims);
      check_trt(ok, "setInputShape failed");
    }
  }

  dump_io_tensors(engine_, context_);

  {
    const auto out_dims = context_->getTensorShape(kwependOutputTensorName);

    output_dtype_ = engine_->getTensorDataType(kwependOutputTensorName);
    output_numel_ = dims_volume(out_dims);
    output_bytes_ = output_numel_ * dtype_size(output_dtype_);

    std::cout
      << "[WEPEND_TRT] output dims=" << dims_to_string(out_dims)
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
        std::cout << "[WEPEND_TRT] layout = [1, attrs, boxes]" << std::endl;
      } else {
        num_boxes = out_dims.d[1];
        num_attrs = out_dims.d[2];
        std::cout << "[WEPEND_TRT] layout = [1, boxes, attrs]" << std::endl;
      }
    } else if (out_dims.nbDims == 2) {
      if (out_dims.d[0] > out_dims.d[1]) {
        num_boxes = out_dims.d[0];
        num_attrs = out_dims.d[1];
        std::cout << "[WEPEND_TRT] layout = [boxes, attrs]" << std::endl;
      } else {
        num_attrs = out_dims.d[0];
        num_boxes = out_dims.d[1];
        std::cout << "[WEPEND_TRT] layout = [attrs, boxes]" << std::endl;
      }
    } else {
      throw std::runtime_error("unsupported output dims: " + dims_to_string(out_dims));
    }

    if (num_attrs < 4 + kwependNumClass) {
      throw std::runtime_error(
        "unexpected output attrs=" + std::to_string(num_attrs) +
        ", expected at least " + std::to_string(4 + kwependNumClass));
    }

    model_bboxes_ = num_boxes;
    decode_mode_ = WependTrtDecodeMode::OnnxXYWH;

    std::cout
      << "[WEPEND_TRT] bboxes=" << num_boxes
      << ", attrs=" << num_attrs
      << ", decode_mode=OnnxXYWH"
      << std::endl;
  }

  CUDA_CHECK(cudaStreamCreate(&stream_));

  cuda_preprocess_init(preprocess_ctx_, kwependMaxInputImageSize);


  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&input_buffer_device_),
    kwependBatchSize * 3 * kwependInputH * kwependInputW * sizeof(float)));

  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&output_buffer_device_),
    output_bytes_));

  decode_ptr_host_ = new float[1 + kwependMaxNumOutputBbox * bbox_element];
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void **>(&decode_ptr_device_),
    sizeof(float) * (1 + kwependMaxNumOutputBbox * bbox_element)));
}

void WependTrt::release()
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

std::list<WependObject> WependTrt::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    return {};
  }

  tmp_img_ = raw_img;

  cv::Mat bgr_img = raw_img.clone();

  cuda_preprocess(
  preprocess_ctx_,
  bgr_img.data,
  bgr_img.cols,
  bgr_img.rows,
  input_buffer_device_,
  kwependInputW,
  kwependInputH,
  stream_);

  bool ok = context_->setInputTensorAddress(
    kwependInputTensorName,
    static_cast<void *>(input_buffer_device_));
  check_trt(ok, "TensorRT setInputTensorAddress failed");

  ok = context_->setTensorAddress(
    kwependOutputTensorName,
    static_cast<void *>(output_buffer_device_));
  check_trt(ok, "TensorRT setTensorAddress failed");

  ok = context_->enqueueV3(stream_);
  check_trt(ok, "TensorRT enqueueV3 failed");

  CUDA_CHECK(cudaMemsetAsync(
    decode_ptr_device_,
    0,
    sizeof(float) * (1 + kwependMaxNumOutputBbox * bbox_element),
    stream_));

  if (decode_mode_ == WependTrtDecodeMode::OnnxXYWH) {
    cuda_decode_onnx_xywh(
      static_cast<const float *>(output_buffer_device_),
      model_bboxes_,
      kwependNumClass,
      kwependConfThresh,
      decode_ptr_device_,
      kwependMaxNumOutputBbox,
      stream_);
  } else if (decode_mode_ == WependTrtDecodeMode::RawLTRB) {
    cuda_decode_raw_head(
      static_cast<const float *>(output_buffer_device_),
      model_bboxes_,
      kwependInputW,
      kwependInputH,
      kwependNumClass,
      kwependConfThresh,
      decode_ptr_device_,
      kwependMaxNumOutputBbox,
      stream_);
  } else {
    cuda_decode(
      static_cast<float *>(output_buffer_device_),
      model_bboxes_,
      kwependConfThresh,
      decode_ptr_device_,
      kwependMaxNumOutputBbox,
      stream_);
  }

  cuda_nms(
    decode_ptr_device_,
    kwependNmsThresh,
    kwependMaxNumOutputBbox,
    stream_);

  CUDA_CHECK(cudaMemcpyAsync(
    decode_ptr_host_,
    decode_ptr_device_,
    sizeof(float) * (1 + kwependMaxNumOutputBbox * bbox_element),
    cudaMemcpyDeviceToHost,
    stream_));

  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return parse_trt_output(decode_ptr_host_, bgr_img, frame_count);
}

std::list<WependObject> WependTrt::parse_trt_output(
  const float * decode_ptr_host,
  const cv::Mat & bgr_img,
  int frame_count)
{
  std::list<WependObject> objs;

  if (decode_ptr_host == nullptr) {
    return objs;
  }

  const int count = std::min(
    static_cast<int>(decode_ptr_host[0]),
    kwependMaxNumOutputBbox);

  const auto img_center = get_image_center(bgr_img);

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
    bbox[0] = decode_ptr_host[base + 0];
    bbox[1] = decode_ptr_host[base + 1];
    bbox[2] = decode_ptr_host[base + 2];
    bbox[3] = decode_ptr_host[base + 3];

    cv::Rect local_box = get_wepend_rect(bgr_img, bbox);
    local_box = clamp_rect_to_image(local_box, bgr_img);

    if (local_box.width <= 0 || local_box.height <= 0) {
      continue;
    }

    WependObject obj;
    obj.class_id = class_id;
    obj.confidence = conf;
    obj.name = get_name(class_id);
    obj.box = local_box;
    obj.center = get_center(obj.box);
    obj.offset_x = obj.center.x - img_center.x;
    obj.offset_y = obj.center.y - img_center.y;

    if (!check_name(obj)) {
      continue;
    }
    if (!check_confidence(obj)) {
      continue;
    }

    objs.emplace_back(std::move(obj));
  }

  if (debug_) {
    draw_detections(bgr_img, objs, frame_count);
  }

  return objs;
}

bool WependTrt::check_name(const WependObject & obj) const
{
  return obj.class_id >= 0 && obj.class_id < kwependNumClass;
}

bool WependTrt::check_confidence(const WependObject & obj) const
{
  return obj.confidence >= static_cast<float>(min_confidence_);
}

cv::Point2f WependTrt::get_center(const cv::Rect & box) const
{
  return {
    box.x + box.width * 0.5f,
    box.y + box.height * 0.5f
  };
}

cv::Point2f WependTrt::get_image_center(const cv::Mat & img) const
{
  return {
    img.cols * 0.5f,
    img.rows * 0.5f
  };
}

std::string WependTrt::get_name(int class_id) const
{
  switch (class_id) {
    case 0: return "spear";
    case 1: return "fist";
    case 2: return "palm";
    default: return "unknown";
  }
}

void WependTrt::draw_detections(
  const cv::Mat & img,
  const std::list<WependObject> & objs,
  int frame_count) const
{
  cv::Mat vis = img.clone();
  const auto img_center = get_image_center(img);

  cv::putText(
    vis,
    fmt::format("[{}]", frame_count),
    cv::Point(10, 30),
    cv::FONT_HERSHEY_SIMPLEX,
    1.0,
    cv::Scalar(255, 255, 255),
    2);

  cv::circle(vis, img_center, 5, cv::Scalar(255, 255, 0), -1);

  for (const auto & obj : objs) {
    cv::rectangle(vis, obj.box, cv::Scalar(0, 255, 0), 2);
    cv::circle(vis, obj.center, 5, cv::Scalar(0, 0, 255), -1);

    const auto label = fmt::format("{} {:.2f}", obj.name, obj.confidence);

    cv::Point text_pos(obj.box.x, obj.box.y - 5);
    if (text_pos.y < 0) {
      text_pos.y = 10;
    }

    cv::putText(
      vis,
      label,
      text_pos,
      cv::FONT_HERSHEY_SIMPLEX,
      0.6,
      cv::Scalar(0, 255, 0),
      2);

    const auto offset_text = fmt::format("dx:{:.1f} dy:{:.1f}", obj.offset_x, obj.offset_y);
    cv::putText(
      vis,
      offset_text,
      obj.center,
      cv::FONT_HERSHEY_SIMPLEX,
      0.6,
      cv::Scalar(0, 255, 0),
      2);
  }

  cv::imshow("wepend_trt", vis);
}

}  // namespace detector