#include "model.h"

#define USE_FP16 // set USE_INT8 or USE_FP16 or USE_FP32

Model::Model(std::string file_path, unsigned int max_batch_size)
    : file_path_(file_path), max_batch_size_(max_batch_size) {};

// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> Model::LoadWeights() {
  std::cout << "Loading weights: " << this->file_path_ << std::endl;
  std::map<std::string, nvinfer1::Weights> weight_map;

  // Open Weight file
  std::ifstream input_file(this->file_path_);
  assert(input_file.is_open() && "Unable to load weight file. please check if the .wts file path is right!");

  // Read number of weight blobs
  int32_t count;
  input_file >> count;
  assert(count > 0 && "Invalid weight map file.");

  while (count--) {
    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
    uint32_t size;

    // Read name and type of blob
    std::string name;
    input_file >> name >> std::dec >> size;
    wt.type = nvinfer1::DataType::kFLOAT;

    // Load blob
    uint32_t *val = new uint32_t[size];
    for (uint32_t x = 0, y = size; x < y; ++x) {
      input_file >> std::hex >> val[x];
    }
    wt.values = val;

    wt.count = size;
    weight_map[name] = wt;
  }
  return weight_map;
}

//  DigitModel(
//  (conv1): Conv2d(1, 128, kernel_size=(5, 5), stride=(1, 1))
//  (act1): ReLU()
//      (conv2): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1))
//  (act2): ReLU()
//      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
//  (dropout1): Dropout(p=0.5, inplace=False)
//  (conv3): Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1))
//  (act3): ReLU()
//      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
//  (dropout2): Dropout(p=0.5, inplace=False)
//  (fla1): Flatten(start_dim=1, end_dim=-1)
//  (linear1): Linear(in_features=576, out_features=512, bias=True)
//  (act4): ReLU()
//      (dropout3): Dropout(p=0.5, inplace=False)
//  (linear2): Linear(in_features=512, out_features=256, bias=True)
//  (act5): ReLU()
//      (dropout4): Dropout(p=0.5, inplace=False)
//  (linear3): Linear(in_features=256, out_features=64, bias=True)
//  (act6): ReLU()
//      (dropout5): Dropout(p=0.5, inplace=False)
//  (linear4): Linear(in_features=64, out_features=10, bias=True)
//  (act7): Softmax(dim=1)
//  )
nvinfer1::ICudaEngine *Model::CreateEngine(nvinfer1::IBuilder *builder,
                                           nvinfer1::IBuilderConfig *config,
                                           nvinfer1::DataType dt) {
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
  // Create input tensor of shape { 1, 1, 28, 28 } with name this->kInputBlobName_
  nvinfer1::ITensor *data = network->addInput(this->kInputBlobName_.c_str(), dt,
                                              nvinfer1::Dims3{1, this->kInputH_, this->kInputW_});
  assert(data);

  std::map<std::string, nvinfer1::Weights> weightMap = LoadWeights();
  nvinfer1::Weights empty_wts{nvinfer1::DataType::kFLOAT, nullptr, 0};

  // Add Conv2d layer with output_size of 128, kernel size of 5x5.
  nvinfer1::IConvolutionLayer *conv1 = network->addConvolutionNd(*data,
                                                                 128,
                                                                 nvinfer1::DimsHW{5, 5},
                                                                 weightMap["conv1.weight"],
                                                                 weightMap["conv1.bias"]);
  assert(conv1);
//  conv1->setStrideNd(nvinfer1::DimsHW{1, 1});

  // Add activation layer using the ReLU algorithm.
  nvinfer1::IActivationLayer *relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu1);

  // Add Conv2d layer with output_size of 128, kernel size of 5x5.
  nvinfer1::IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0),
                                                                 128,
                                                                 nvinfer1::DimsHW{5, 5},
                                                                 weightMap["conv2.weight"],
                                                                 weightMap["conv2.bias"]);
  assert(conv2);
//  conv2->setStrideNd(nvinfer1::DimsHW{1, 1});

  // Add activation layer using the ReLU algorithm.
  nvinfer1::IActivationLayer *relu2 = network->addActivation(*conv2->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu2);

  // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
  // No params for this Dropout layer
  nvinfer1::IPoolingLayer *pool1 = network->addPoolingNd(*relu2->getOutput(0),
                                                         nvinfer1::PoolingType::kMAX,
                                                         nvinfer1::DimsHW{2, 2});
  assert(pool1);
  pool1->setStrideNd(nvinfer1::DimsHW{2, 2});

  // Add Conv2d layer with output_size of 64, kernel size of 5x5.
  nvinfer1::IConvolutionLayer *conv3 = network->addConvolutionNd(*pool1->getOutput(0),
                                                                 64,
                                                                 nvinfer1::DimsHW{5, 5},
                                                                 weightMap["conv3.weight"],
                                                                 weightMap["conv3.bias"]);
  assert(conv3);
//  conv3->setStrideNd(nvinfer1::DimsHW{1, 1});

  // Add activation layer using the ReLU algorithm.
  nvinfer1::IActivationLayer *relu3 = network->addActivation(*conv3->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu3);

  // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
  // No params for this Dropout layer
  nvinfer1::IPoolingLayer *pool2 = network->addPoolingNd(*relu3->getOutput(0),
                                                         nvinfer1::PoolingType::kMAX,
                                                         nvinfer1::DimsHW{2, 2});
  assert(pool2);
  pool2->setStrideNd(nvinfer1::DimsHW{2, 2});

  // Currently, TensorRT only supports flatten layer which is placed in front of FullyConnected layers.
  // In this case, TensorRT implicitly flattens the input and no extra layer needs to be added.
  nvinfer1::IFullyConnectedLayer *fc1 = network->addFullyConnected(*pool2->getOutput(0),
                                                                   512,
                                                                   weightMap["linear1.weight"],
                                                                   weightMap["linear1.bias"]);
  assert(fc1);

  nvinfer1::IActivationLayer *relu4 = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu4);

  nvinfer1::IFullyConnectedLayer *fc2 = network->addFullyConnected(*relu4->getOutput(0),
                                                                   256,
                                                                   weightMap["linear2.weight"],
                                                                   weightMap["linear2.bias"]);
  assert(fc2);

  nvinfer1::IActivationLayer *relu5 = network->addActivation(*fc2->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu5);

  nvinfer1::IFullyConnectedLayer *fc3 = network->addFullyConnected(*relu5->getOutput(0),
                                                                   64,
                                                                   weightMap["linear3.weight"],
                                                                   weightMap["linear3.bias"]);
  assert(fc3);

  nvinfer1::IActivationLayer *relu6 = network->addActivation(*fc3->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu6);

  nvinfer1::IFullyConnectedLayer *fc4 = network->addFullyConnected(*relu6->getOutput(0),
                                                                   10,
                                                                   weightMap["linear4.weight"],
                                                                   weightMap["linear4.bias"]);
  assert(fc4);

  // Add softmax layer.
  nvinfer1::ISoftMaxLayer *softmax1 = network->addSoftMax(*fc4->getOutput(0));
  assert(softmax1);

  softmax1->getOutput(0)->setName(this->kOutputBlobName_.c_str());
  std::cout << "set name out" << std::endl;
  network->markOutput(*softmax1->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(this->max_batch_size_);
  config->setMaxWorkspaceSize(1 * (1 << 20));  // 1MB

#if defined(USE_FP16)
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
  std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
  assert(builder->platformHasFastInt8());
  config->setFlag(nvinfer1::BuilderFlag::kINT8);
  nvinfer1::Int8EntropyCalibrator2* calibrator = new nvinfer1::IInt8EntropyCalibrator2();
  config->setInt8Calibrator(calibrator);
#endif

  std::cout << "Building engine, please wait for a while..." << std::endl;
  nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto &mem : weightMap) {
    free((void *) (mem.second.values));
  }

  return engine;
}

void Model::ApiToModel(nvinfer1::IHostMemory **model_stream) {
  // Create builder
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(this->logger_);
  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  nvinfer1::ICudaEngine *engine = CreateEngine(builder, config, nvinfer1::DataType::kFLOAT);
  assert(nullptr != engine);

  // Serialize the engine
  (*model_stream) = engine->serialize();

  // Close everything
  engine->destroy();
  builder->destroy();
}

void Model::DoInference(float *input, float *output, int batch_size) {
  const nvinfer1::ICudaEngine &engine = this->context_->getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(2 == engine.getNbBindings());
  void *buffers[2];

  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int input_index = engine.getBindingIndex(this->kInputBlobName_.c_str());
  const int output_index = engine.getBindingIndex(this->kOutputBlobName_.c_str());

  // Create GPU buffers on device
  this->CudaCheck(cudaMalloc(&buffers[input_index], batch_size * 1 * this->kInputH_ * this->kInputW_ * sizeof(float)));
  this->CudaCheck(cudaMalloc(&buffers[output_index], batch_size * this->kOutputSize_ * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  this->CudaCheck(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
  this->CudaCheck(cudaMemcpyAsync(buffers[input_index],
                                  input,
                                  batch_size * 1 * this->kInputH_ * this->kInputW_ * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream));
  this->context_->enqueue(batch_size, buffers, stream, nullptr);
  this->CudaCheck(cudaMemcpyAsync(output,
                                  buffers[output_index],
                                  batch_size * this->kOutputSize_ * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  this->CudaCheck(cudaFree(buffers[input_index]));
  this->CudaCheck(cudaFree(buffers[output_index]));
}

bool Model::WriteEngine(std::string file_name) {
  nvinfer1::IHostMemory *model_stream{nullptr};
  this->ApiToModel(&model_stream);
  assert(nullptr != model_stream);

  std::ofstream p(file_name);
  if (!p) {
    return false;
  }
  p.write(reinterpret_cast<const char *>(model_stream->data()), model_stream->size());
  model_stream->destroy();
  return true;
}

void Model::ReadEngine(std::string file_name) {
  std::ifstream file(file_name, std::ios::binary);
  if (file.good()) {
    file.seekg(0, file.end);
    this->size_ = file.tellg();
    file.seekg(0, file.beg);
    this->trt_model_stream_ = new char[this->size_];
    assert(this->trt_model_stream_);
    file.read(this->trt_model_stream_, this->size_);
    file.close();
  }
  this->runtime_ = nvinfer1::createInferRuntime(this->logger_);
  assert(nullptr != this->runtime_);
  this->engine_ = this->runtime_->deserializeCudaEngine(this->trt_model_stream_, this->size_, nullptr);
  assert(nullptr != this->engine_);
  this->context_ = this->engine_->createExecutionContext();
  assert(nullptr != this->context_);
}

int Model::InferenceOneImg(float *data) {
  // Run inference
  float prob[this->kOutputSize_];
  this->DoInference(data, prob, 1);

  for (int i = 0; i < this->kOutputSize_; ++i) {
    if (!this->IsZero(prob[i])) {
      return i;
    }
  }
  return -1;
}

std::vector<int> Model::InferenceBatchImg(float *data, std::vector<int> &prob_vec, int batch_size) {
  bool s = false;
  prob_vec.clear();
  // Run inference
  float prob[batch_size * this->kOutputSize_];
  this->DoInference(data, prob, batch_size);

  for (int i = 0; i < batch_size; ++i) {
//    for (int j = 0; j < this->kOutputSize_; ++j) {
//      if (!this->IsZero(prob[i * this->kOutputSize_ + j])) {
//        prob_vec.push_back(j);
//        s = true;
//        break;
//      }
//    }
    prob_vec.push_back(this->MaxNum(&prob[i * this->kOutputSize_]));
  }
  return prob_vec;
}

Model::~Model() {
  // todo:
  if (this->trt_model_stream_) {
    delete this->trt_model_stream_;
  }
  // Destroy the engine
  if (this->context_) {
    this->context_->destroy();
  }
  if (this->engine_) {
    this->engine_->destroy();
  }
  if (this->runtime_) {
    this->runtime_->destroy();
  }
}
