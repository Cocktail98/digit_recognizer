#ifndef CPP_TENSORRT__MODEL_H_
#define CPP_TENSORRT__MODEL_H_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <map>
#include <chrono>

#include "logging.h"

static const int kInputH = 224;
static const int kInputW = 224;
static const int kOutputSize = 1000;

static const char *kInputBlobName = "data";
static const char *kOutputBlobName = "prob";

class Model {
 public:
  explicit Model(std::string file_path);
  std::map<std::string, nvinfer1::Weights> LoadWeights();

 private:
  static Logger logger_;
  std::string file_path_;
  void Check(cudaError_t error);
};

#endif //CPP_TENSORRT__MODEL_H_
