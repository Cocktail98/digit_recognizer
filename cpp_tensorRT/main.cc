#include <iostream>
#include <chrono>
#include <string>
#include <ctime>

#include "model.h"
#include "utils/data_loader.h"
#include "utils/print_time.h"

// Maximum allowed batch_size
static const int kMaxBatchSize = 1024;
// Batch_size for inference
static const int kBatchSize = 1024;
static const std::string kDataFilePath = "../../data/test.csv";
static const std::string kOutFilePath = "../submission.csv";

int main(int argc, char **argv) {
  if (2 != argc) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./main -s   // serialize model to plan file" << std::endl;
    std::cerr << "./main -d   // deserialize plan file and run inference" << std::endl;
    return -1;
  }

  Model model("../cnn.wts", kMaxBatchSize);

  if ("-s" == std::string(argv[1])) {
    if (model.WriteEngine("cnn.engine")) {
      return 1;
    } else {
      std::cerr << "could not open plan output file" << std::endl;
      return -1;
    }
  } else if ("-d" == std::string(argv[1])) {
    model.ReadEngine("cnn.engine");
  } else {
    return -1;
  }

  std::ofstream out_file;
  out_file.open(kOutFilePath, std::ios::out);
  out_file << "ImageId" << ',' << "Label" << std::endl;

  DataLoader data_loader(kDataFilePath);
  // display digit by num
  // data_loader.ShowDigitByNum(0);
  auto start = std::chrono::system_clock::now();

  std::cout << std::endl;
  std::vector<int> res_vec;
  for (int i = 0, l = data_loader.GetDataSize();; ++i) {
    if (0 == l / kBatchSize) {
      float *data_arr = data_loader.GetDigitsData(i * kBatchSize, -1);
      model.InferenceBatchImg(data_arr, res_vec, l);
      for (int j = 0; j < res_vec.size(); ++j) {
        out_file << (i * kBatchSize + j + 1) << ',' << res_vec[j] << std::endl;
      }
      break;
    } else {
      float *data_arr = data_loader.GetDigitsData(i * kBatchSize, (i + 1) * kBatchSize);
      model.InferenceBatchImg(data_arr, res_vec, kBatchSize);
      for (int j = 0; j < kBatchSize; ++j) {
        out_file << i * kBatchSize + j + 1 << ',' << res_vec[j] << std::endl;
      }
      l -= kBatchSize;
    }
  }
  PrintTime();
  auto end = std::chrono::system_clock::now();
  std::cout << "Inference is finished, total time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << "ms" << std::endl;

//  std::cout << model.InferenceOneImg(&data_arr[1538]);

  // close file_stream
  out_file.close();

  return 0;
}
