#pragma once
// 推理精度选择
#define INFOR_USE_FP16          // 使用 FP16 半精度推理，速度更快，显存占用更低
//#define INFOR_USE_FP32        // 使用 FP32 单精度推理，结果更稳，便于排查精度问题
// #define INFOR_USE_INT8       // 使用 INT8 量化推理，需要校准数据，速度更快但部署更复杂

// =========================
// TensorRT 输入输出张量名称
// 需要和 engine / onnx 里的真实名字一致
// =========================
const static char* kInforInputTensorName = "images";   // 输入张量名称
const static char* kInforOutputTensorName = "output0"; // 检测输出张量名称
const static char* kInforProtoTensorName = "proto";    // 分割任务中的 mask 原型图张量名称

// =========================
// 检测 / 关键点 / OBB 相关类别参数
// =========================
const static int kInforNumClass = 14;       // infor 类别数：pingmu, 0-9, A, C, E
const static int kInforPoseNumClass = 1;    // 姿态模型类别数，当前 infor 检测不使用
const static int kInforNumberOfPoints = 17; // 关键点总数，当前 infor 检测不使用

// OBB（旋转框）模型的类别数，当前 infor 检测不使用
constexpr static int kInforObbNumClass = 15;

// OBB 额外参数个数，当前 infor 检测不使用
const static int kInforObbNe = 1;

// =========================
// 推理批大小 / GPU 设备
// =========================
const static int kInforBatchSize = 1;  // batch 大小，当前按单张图推理
const static int kInforGpuId = 0;      // 使用的 GPU 编号，0 表示第一块 GPU

// =========================
// 普通检测模型输入尺寸
// 需要和训练/导出 ONNX/构建 engine 时一致
// =========================
const static int kInforInputH = 640;   // 输入高度
const static int kInforInputW = 640;   // 输入宽度

// =========================
// OBB（旋转框）模型输入尺寸
// 仅 OBB 任务使用，当前 infor 检测不使用
// =========================
const static int kInforObbInputH = 1024;
const static int kInforObbInputW = 1024;

// =========================
// 后处理阈值
// =========================
const static float kInforNmsThresh = 0.5f;           // NMS 阈值，越大保留框越多
const static float kInforConfThresh = 0.5f;          // 目标置信度阈值，低于这个值的框会被过滤
const static float kInforConfThreshKeypoints = 0.5f; // 关键点置信度阈值，当前 infor 检测不使用

// =========================
// 输入图像与输出框数量限制
// =========================
const static int kInforMaxInputImageSize = 3000 * 3000; // 预处理时允许的最大输入图像像素数
const static int kInforMaxNumOutputBbox = 1000;         // 最多保留的检测框数量

// =========================
// INT8 量化时的校准图片目录
// 只有启用 INT8 时才会用到
// =========================
const static char* kInforInputQuantizationFolder = "./infor_calib";