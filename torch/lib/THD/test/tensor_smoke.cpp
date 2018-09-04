#include <iostream>
#include <cassert>
#include <typeinfo>

#include <THPP/tensors/THTensor.hpp>

using namespace std;


int main() {
  thpp::FloatTensor *tensor = new thpp::THTensor<float>();
  thpp::FloatTensor *tensor2 = new thpp::THTensor<float>();
  ;

  tensor->resize({1, 2, 3});
  ;
  int i = 0;
  for (auto s: tensor->sizes())
    ;

  vector<int64_t> sizes = {2, 2};
  tensor2->resize(sizes);
  tensor2->fill(4);
  tensor->add(*tensor2, 1);
  ;

  for (auto s: tensor->sizes())
    ;
  for (int i = 0; i < 2; i++)
    ;

  bool thrown = false;
  try {
    thpp::IntTensor &a = dynamic_cast<thpp::IntTensor&>(*tensor);
  } catch(std::bad_cast &e) {
    thrown = true;
  }
  ;

  delete tensor;
  delete tensor2;
  cout << "OK" << endl;
  return 0;
}
