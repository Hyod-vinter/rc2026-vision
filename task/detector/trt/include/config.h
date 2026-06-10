#pragma once
// 推理精度选择
#define USE_FP16          // 使用 FP16 半精度推理，速度更快，显存占用更低
//#define USE_FP32        // 使用 FP32 单精度推理，结果更稳，便于排查精度问题
// #define USE_INT8       // 使用 INT8 量化推理，需要校准数据，速度更快但部署更复杂

// =========================
// TensorRT 输入输出张量名称
// 需要和 engine / onnx 里的真实名字一致
// =========================
const static char* kInputTensorName = "images";   // 输入张量名称
const static char* kOutputTensorName = "output0"; // 检测输出张量名称
const static char* kProtoTensorName = "proto";    // 分割任务中的 mask 原型图张量名称

// =========================
// 检测 / 关键点 / OBB 相关类别参数
// =========================
const static int kNumClass = 32;        // 普通目标检测类别数，比如你当前模型是 32 类
const static int kPoseNumClass = 1;     // 姿态模型类别数，很多 pose 模型只有 1 类（人）
const static int kNumberOfPoints = 17;  // 关键点总数，常见人体关键点是 17 个

// OBB（旋转框）模型的类别数
constexpr static int kObbNumClass = 15;

// OBB 额外参数个数，通常表示角度等附加信息
const static int kObbNe = 1;

// =========================
// 推理批大小 / GPU 设备
// =========================
const static int kBatchSize = 1;  // batch 大小，当前按单张图推理
const static int kGpuId = 0;      // 使用的 GPU 编号，0 表示第一块 GPU

// =========================
// 普通检测模型输入尺寸
// 需要和训练/导出 ONNX/构建 engine 时一致
// =========================
const static int kInputH = 640;   // 输入高度
const static int kInputW = 640;   // 输入宽度

// =========================
// OBB（旋转框）模型输入尺寸
// 仅 OBB 任务使用
// =========================
const static int kObbInputH = 1024;  // OBB 输入高度
const static int kObbInputW = 1024;  // OBB 输入宽度

// =========================
// 后处理阈值
// =========================
const static float kNmsThresh = 0.5f;           // NMS 阈值，越大保留框越多
const static float kConfThresh = 0.5f;           // 目标置信度阈值，低于这个值的框会被过滤
const static float kConfThreshKeypoints = 0.5f;  // 关键点置信度阈值

// =========================
// 输入图像与输出框数量限制
// =========================
const static int kMaxInputImageSize = 3000 * 3000; // 预处理时允许的最大输入图像像素数
const static int kMaxNumOutputBbox = 1000;         // 最多保留的检测框数量

// =========================
// INT8 量化时的校准图片目录
// 只有启用 INT8 时才会用到
// =========================
const static char* kInputQuantizationFolder = "./coco_calib";

// =========================
// 分类模型参数
// 如果当前不是分类任务，可以先不管
// =========================
constexpr static int kClsNumClass = 1000; // 分类模型类别数
constexpr static int kClsInputH = 224;    // 分类模型输入高度
constexpr static int kClsInputW = 224;    // 分类模型输入宽度

