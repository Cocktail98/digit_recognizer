#include "model.h"

Model::Model(std::string file_path) : file_path_(file_path) {};

// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> Model::LoadWeights() {
  std::cout << "Loading weights: " << this->file_path_ << std::endl;
  std::map<std::string, nvinfer1::Weights> weight_map;

  // Open Weight file
  std::ifstream input_file(this->file_path_);
  assert(input_file.is_open() && "Unable to load weight file.");

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

void Model::Check(cudaError_t error) {
  do {
    auto ret = (error);
    if (ret != 0) {
      std::cerr << "Cuda failure: " << ret << std::endl;
      abort();
    }
  } while (0);
}

