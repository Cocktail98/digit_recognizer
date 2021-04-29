#include "alexnet.h"

// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> LoadWeights(const std::string file) {
  std::cout << "Loading weights: " << file << std::endl;
  std::map<std::string, nvinfer1::Weights> weightMap;

  // Open Weight file
  std::ifstream input_file(file);
  assert(input_file.is_open() && "Unable to load weight file.");

  // Read number of weight blobs
  int32_t count;
  input_file >> count;
  assert(count > 0 && "Invalid weight map file.");
//  std::cout << "cout: " << count << std::endl;

  while (count--) {
    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
    uint32_t size;

    // Read name and type of blob
    std::string name;
    input_file >> name >> std::dec >> size;
//    std::cout << "layer name: " << name << std::endl;
    wt.type = nvinfer1::DataType::kFLOAT;

    // Load blob
    uint32_t *val = new uint32_t[size];
    for (uint32_t x = 0, y = size; x < y; ++x) {
      input_file >> std::hex >> val[x];
    }
    wt.values = val;

    wt.count = size;
    weightMap[name] = wt;
  }
  return weightMap;
}

nvinfer1::ICudaEngine *CreateEngine(unsigned int max_batch_size,
                                    const std::string weight_file_path,
                                    nvinfer1::IBuilder *builder,
                                    nvinfer1::IBuilderConfig *config,
                                    nvinfer1::DataType dt) {
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);

  // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
  nvinfer1::ITensor *data = network->addInput(kInputBlobName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
  assert(data);

  std::map<std::string, nvinfer1::Weights> weightMap = LoadWeights(weight_file_path);
  nvinfer1::Weights empty_wts{nvinfer1::DataType::kFLOAT, nullptr, 0};

  nvinfer1::IConvolutionLayer *conv1 = network->addConvolutionNd(*data,
                                                                 64,
                                                                 nvinfer1::DimsHW{11, 11},
                                                                 weightMap["features.0.weight"],
                                                                 weightMap["features.0.bias"]);
  assert(conv1);
  conv1->setStrideNd(nvinfer1::DimsHW{4, 4});
  conv1->setPaddingNd(nvinfer1::DimsHW{2, 2});

  // Add activation layer using the ReLU algorithm.
  nvinfer1::IActivationLayer *relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu1);

  // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
  nvinfer1::IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0),
                                                         nvinfer1::PoolingType::kMAX,
                                                         nvinfer1::DimsHW{3, 3});
  assert(pool1);
  pool1->setStrideNd(nvinfer1::DimsHW{2, 2});

  nvinfer1::IConvolutionLayer *conv2 = network->addConvolutionNd(*pool1->getOutput(0),
                                                                 192,
                                                                 nvinfer1::DimsHW{5, 5},
                                                                 weightMap["features.3.weight"],
                                                                 weightMap["features.3.bias"]);
  assert(conv2);
  conv2->setPaddingNd(nvinfer1::DimsHW{2, 2});

  nvinfer1::IActivationLayer *relu2 = network->addActivation(*conv2->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu2);

  nvinfer1::IPoolingLayer *pool2 = network->addPoolingNd(*relu2->getOutput(0),
                                                         nvinfer1::PoolingType::kMAX,
                                                         nvinfer1::DimsHW{3, 3});
  assert(pool2);
  pool2->setStrideNd(nvinfer1::DimsHW{2, 2});

  nvinfer1::IConvolutionLayer *conv3 = network->addConvolutionNd(*pool2->getOutput(0),
                                                                 384,
                                                                 nvinfer1::DimsHW{3, 3},
                                                                 weightMap["features.6.weight"],
                                                                 weightMap["features.6.bias"]);
  assert(conv3);
  conv3->setPaddingNd(nvinfer1::DimsHW{1, 1});

  nvinfer1::IActivationLayer *relu3 = network->addActivation(*conv3->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu3);

  nvinfer1::IConvolutionLayer *conv4 = network->addConvolutionNd(*relu3->getOutput(0),
                                                                 256,
                                                                 nvinfer1::DimsHW{3, 3},
                                                                 weightMap["features.8.weight"],
                                                                 weightMap["features.8.bias"]);
  assert(conv4);
  conv4->setPaddingNd(nvinfer1::DimsHW{1, 1});

  nvinfer1::IActivationLayer *relu4 = network->addActivation(*conv4->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu4);

  nvinfer1::IConvolutionLayer *conv5 = network->addConvolutionNd(*relu4->getOutput(0),
                                                                 256,
                                                                 nvinfer1::DimsHW{3, 3},
                                                                 weightMap["features.10.weight"],
                                                                 weightMap["features.10.bias"]);
  assert(conv5);
  conv5->setPaddingNd(nvinfer1::DimsHW{1, 1});

  nvinfer1::IActivationLayer *relu5 = network->addActivation(*conv5->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu5);

  nvinfer1::IPoolingLayer *pool3 = network->addPoolingNd(*relu5->getOutput(0),
                                                         nvinfer1::PoolingType::kMAX,
                                                         nvinfer1::DimsHW{3, 3});
  assert(pool3);
  pool3->setStrideNd(nvinfer1::DimsHW{2, 2});

  nvinfer1::IFullyConnectedLayer *fc1 = network->addFullyConnected(*pool3->getOutput(0),
                                                                   4096,
                                                                   weightMap["classifier.1.weight"],
                                                                   weightMap["classifier.1.bias"]);
  assert(fc1);

  nvinfer1::IActivationLayer *relu6 = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu6);

  nvinfer1::IFullyConnectedLayer *fc2 = network->addFullyConnected(*relu6->getOutput(0),
                                                                   4096,
                                                                   weightMap["classifier.4.weight"],
                                                                   weightMap["classifier.4.bias"]);
  assert(fc2);

  nvinfer1::IActivationLayer *relu7 = network->addActivation(*fc2->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(relu7);

  nvinfer1::IFullyConnectedLayer *fc3 = network->addFullyConnected(*relu7->getOutput(0),
                                                                   1000,
                                                                   weightMap["classifier.6.weight"],
                                                                   weightMap["classifier.6.bias"]);
  assert(fc3);

  fc3->getOutput(0)->setName(kOutputBlobName);
  std::cout << "set name out" << std::endl;
  network->markOutput(*fc3->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(1 << 20);
  nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "build out" << std::endl;

  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto &mem : weightMap) {
    free((void *) (mem.second.values));
  }

  return engine;
}

void doInference(nvinfer1::IExecutionContext &context, float *input, float *output, int batchSize) {
  const nvinfer1::ICudaEngine &engine = context.getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 2);
  void *buffers[2];

  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(kInputBlobName);
  const int outputIndex = engine.getBindingIndex(kOutputBlobName);

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * kInputH * kInputW * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], batchSize * kOutputSize * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex],
                        input,
                        batchSize * 3 * kInputH * kInputW * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream));
  context.enqueue(batchSize, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output,
                        buffers[outputIndex],
                        batchSize * kOutputSize * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}

void APIToModel(unsigned int max_batch_size, const std::string weight_file_path, nvinfer1::IHostMemory **model_stream) {
  // Create builder
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  nvinfer1::ICudaEngine *engine = CreateEngine(max_batch_size, weight_file_path,
                                               builder, config,
                                               nvinfer1::DataType::kFLOAT);
  assert(engine != nullptr);

  // Serialize the engine
  (*model_stream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}