#ifndef CPP_TENSORRT_UTILS_DATA_LOADER_H_
#define CPP_TENSORRT_UTILS_DATA_LOADER_H_

#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <iostream>

class DataLoader {
 public:
  explicit DataLoader(std::string file_path);
  void ShowDigitByNum(const int num);
  float *GetOneDigitDataByNum(const int num);
  float *GetDigitsData(const int begin, const int end);
  int GetDataSize();
 private:
  int each_img_size_;
  int arr_size_;
  std::string file_path_;
  std::vector<std::vector<float>> data_arr_;
};

#endif //CPP_TENSORRT_UTILS_DATA_LOADER_H_
