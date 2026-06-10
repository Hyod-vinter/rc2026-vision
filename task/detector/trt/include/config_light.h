#pragma once

#define INFOR_USE_FP16
// #define INFOR_USE_FP32
// #define INFOR_USE_INT8

// =========================
// TensorRT 输入输出张量名称
// 需要和 engine / onnx 里的真实名字一致
// =========================
const static char* klightInputTensorName = "images";
const static char* klightOutputTensorName = "output0";

// =========================
// 检测类别参数
// 类别确认：0=red, 1=green, 2=blue, 3=yellow, 4=purple, 5=cyan
// 注意：unknown 只作为 light.hpp 中的兜底枚举，不计入模型输出类别数
// =========================
const static int klightNumClass = 6;

// =========================
// 推理批大小 / GPU 设备
// =========================
const static int klightBatchSize = 1;
const static int klightGpuId = 0;

// =========================
// 模型输入尺寸
// 需要和训练 / 导出 ONNX / 构建 engine 时一致
// =========================
const static int klightInputH = 640;
const static int klightInputW = 640;

// =========================
// 后处理阈值
// =========================
const static float klightNmsThresh = 0.5f;
const static float klightConfThresh = 0.7f;

// =========================
// 输入图像与输出框数量限制
// =========================
const static int klightMaxInputImageSize = 3000 * 3000;
const static int klightMaxNumOutputBbox = 1000;

// =========================
// INT8 量化时的校准图片目录
// 只有启用 INT8 时才会用到
// =========================
const static char* klightInputQuantizationFolder = "./light_calib";
