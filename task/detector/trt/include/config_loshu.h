#pragma once

// yolo11_lstrt / LoShu keypoint TensorRT config
static constexpr int kLoShuKptNum = 10;
static constexpr int kLoShuAttrNum = 5 + kLoShuKptNum * 3;      // 35
static constexpr int kLoShuElement = 7 + kLoShuKptNum * 3;      // 37

static constexpr int kLoShuMaxNumOutputBbox = 1000;
static constexpr int kLoShuOutputSize =
    1 + kLoShuMaxNumOutputBbox * kLoShuElement;

static constexpr int kLoShuInputW = 640;
static constexpr int kLoShuInputH = 640;
static constexpr int kLoShuBatchSize = 1;
static constexpr int kLoShuGpuId = 0;
static constexpr int kLoShuMaxInputImageSize = 3000 * 3000;

static constexpr const char* kLoShuInputTensorName = "images";
static constexpr const char* kLoShuOutputTensorName = "output0";
