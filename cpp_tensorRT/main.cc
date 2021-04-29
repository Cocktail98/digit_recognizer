#include <iostream>

#include "model.h"

int main() {
  Model model("../cnn.wts");
  model.LoadWeights();
  return 0;
}
