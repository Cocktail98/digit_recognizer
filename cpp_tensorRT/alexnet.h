#ifndef ALEXNET__ALEXNET_H_
#define ALEXNET__ALEXNET_H_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <string>
#include <map>
#include <iostream>
#include <cassert>
#include <cstdint>

#include "logging.h"

#define CHECK(status) \
  do {\
    auto ret = (status);\
    if (0 != ret) {\
        std::cerr << "Cuda failure: " << ret << std::endl;\
        abort();\
    }\
  } while (0)

static const int kInputH = 224;
static const int kInputW = 224;
static const int kOutputSize = 1000;

static const char *kInputBlobName = "data";
static const char *kOutputBlobName = "prob";

static Logger gLogger;

std::map<std::string, nvinfer1::Weights> LoadWeights(const std::string file);
// Creat the engine using only the API and not any parser.
nvinfer1::ICudaEngine *CreateEngine(unsigned int max_batch_size,
                                    const std::string weight_file_path,
                                    nvinfer1::IBuilder *builder,
                                    nvinfer1::IBuilderConfig *config,
                                    nvinfer1::DataType dt);
void APIToModel(unsigned int max_batch_size, const std::string weight_file_path, nvinfer1::IHostMemory **model_stream);
void doInference(nvinfer1::IExecutionContext &context, float *input, float *output, int batchSize);

#endif //ALEXNET__ALEXNET_H_
