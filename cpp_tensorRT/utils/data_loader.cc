#include "data_loader.h"

DataLoader::DataLoader(std::string file_path) : file_path_(file_path) {
  std::ifstream input_data(this->file_path_);
  std::string line;
  // Skip the column name and do not process the first line
  getline(input_data, line);
  while (getline(input_data, line)) {
    std::vector<float> data_line;
    std::string num;
    std::istringstream read_str(line);
    for (int j = 0; j < 28 * 28; j++) {
      getline(read_str, num, ',');
      data_line.push_back(atof(num.c_str()) / 255);
    }
    this->data_arr_.push_back(data_line);
  }
  this->arr_size_ = this->data_arr_.size();
  this->each_img_size_ = this->data_arr_[0].size();
}

void DataLoader::ShowDigitByNum(const int num) {
  std::cout << "--------------------------------------------------------" << std::endl;
  for (int i = 0; i < 28 * 28; ++i) {
    if (0 == this->data_arr_[num][i]) {
      std::cout << "  ";
    } else {
      std::cout << "##";
    }
    if (0 == i % 28) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl << "--------------------------------------------------------" << std::endl;
}

float *DataLoader::GetOneDigitDataByNum(const int num) {
  float *arr = new float[this->data_arr_[num].size()];
  std::memcpy(arr, &this->data_arr_[num][0], this->each_img_size_ * sizeof(this->data_arr_[num][0]));
  return arr;
}

float *DataLoader::GetDigitsData(const int begin, const int end) {
//  int data_size = 0;
  int begin_id = 0, end_id = 0;
  // if end=-1, return all data
  if (begin < 0) {
    begin_id = 0;
  } else {
    begin_id = begin;
  }
  if (end < 0) {
    end_id = this->arr_size_;
  } else {
    end_id = end;
  }
  if (end_id < begin_id) {
    return nullptr;
  }

  std::vector<std::vector<float>> arr_vec;
  arr_vec.insert(arr_vec.begin(), this->data_arr_.begin() + begin_id, this->data_arr_.begin() + end_id);

  int data_size = end_id - begin_id;

  float *arr = new float[data_size * this->each_img_size_];
  for (int i = 0; i < data_size; ++i) {
    std::memcpy(&arr[i * this->each_img_size_],
                &arr_vec[i][0],
                this->each_img_size_ * sizeof(arr_vec[i][0]));
  }
  return arr;
}

int DataLoader::GetDataSize() {
  return this->arr_size_;
}