#ifndef CPP_TENSORRT__MODEL_H_
#define CPP_TENSORRT__MODEL_H_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>

#include "logging.h"

class Model {
 public:
  explicit Model(std::string file_path, unsigned int max_batch_size);
  std::map<std::string, nvinfer1::Weights> LoadWeights();
  // Creat the engine using only the API and not any parser.
  nvinfer1::ICudaEngine *CreateEngine(nvinfer1::IBuilder *builder,
                                      nvinfer1::IBuilderConfig *config,
                                      nvinfer1::DataType dt);
  void ApiToModel(nvinfer1::IHostMemory **model_stream);
  void DoInference(float *input, float *output, int batch_size);
  bool WriteEngine(std::string file_name);
  void ReadEngine(std::string file_name);
  int InferenceOneImg(float *data);
  std::vector<int> InferenceBatchImg(float *data, std::vector<int> &prob_vec, int batch_size);
  ~Model();

 private:
  // const data
  const int kInputH_ = 28;
  const int kInputW_ = 28;
  const int kOutputSize_ = 10;
  const std::string kInputBlobName_ = "data";
  const std::string kOutputBlobName_ = "prob";
  Logger logger_;
  std::string file_path_;
  unsigned int max_batch_size_;
  // create a model using the API directly and serialize it to a stream
  char *trt_model_stream_{nullptr};
  size_t size_{0};
  nvinfer1::IRuntime *runtime_;
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *context_;
  void CudaCheck(cudaError_t error_code);
  bool IsZero(float num);
  int MaxNum(float *nums);
};

inline void Model::CudaCheck(cudaError_t error_code) {
  if (cudaSuccess != error_code) {
    std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
    assert(0);
  }
}

inline bool Model::IsZero(float num) {
  if (std::abs(num) <= 1e-6) { return true; }
  else { return false; }
}

inline int Model::MaxNum(float *nums) {
  int max_id = 0;
  for (int i = 0; i < this->kOutputSize_; ++i) {
    if (nums[i] > nums[max_id]) {
      max_id = i;
    }
  }
  return max_id;
}

#endif //CPP_TENSORRT__MODEL_H_
