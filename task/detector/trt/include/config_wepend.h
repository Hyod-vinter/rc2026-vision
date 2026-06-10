#pragma once

#define INFOR_USE_FP16
// #define INFOR_USE_FP32
// #define INFOR_USE_INT8

// =========================
// TensorRT 输入输出张量名称
// 需要和 engine / onnx 里的真实名字一致
// =========================
const static char* kwependInputTensorName = "images";
const static char* kwependOutputTensorName = "output0";

// =========================
// 检测类别参数
// 类别确认：0=spear, 1=fist, 2=palm
// =========================
const static int kwependNumClass = 3;

// =========================
// 推理批大小 / GPU 设备
// =========================
const static int kwependBatchSize = 1;
const static int kwependGpuId = 0;

// =========================
// 模型输入尺寸
// 需要和训练 / 导出 ONNX / 构建 engine 时一致
// =========================
const static int kwependInputH = 640;
const static int kwependInputW = 640;

// =========================
// 后处理阈值
// =========================
const static float kwependNmsThresh = 0.5f;
const static float kwependConfThresh = 0.7f;

// =========================
// 输入图像与输出框数量限制
// =========================
const static int kwependMaxInputImageSize = 3000 * 3000;
const static int kwependMaxNumOutputBbox = 1000;

// =========================
// INT8 量化时的校准图片目录
// 只有启用 INT8 时才会用到
// =========================
const static char* kwependInputQuantizationFolder = "./wepend_calib";