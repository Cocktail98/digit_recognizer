#ifndef CPP_TENSORRT_UTILS_PRINT_TIME_H_
#define CPP_TENSORRT_UTILS_PRINT_TIME_H_

#include <ctime>
#include <iostream>

void PrintTime(){
  std::time_t timestamp = std::time(nullptr);
  tm *tm_local = std::localtime(&timestamp);
  std::cout << "[";
  std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
  std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
}

#endif //CPP_TENSORRT_UTILS_PRINT_TIME_H_
