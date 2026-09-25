#include "Data.h"

#include <cstdlib>

namespace core {

void Tok::skip() {
  while (i_ < s_.size()) {
    const char c = s_[i_];
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
      i_ = i_ + 1;
    } else {
      break;
    }
  }
}

bool Tok::has() {
  skip();
  return i_ < s_.size();
}

std::string Tok::next() {
  skip();
  const size_t start = i_;
  while (i_ < s_.size()) {
    const char c = s_[i_];
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
      break;
    }
    i_ = i_ + 1;
  }
  return s_.substr(start, i_ - start);
}

double Tok::num() { return std::atof(next().c_str()); }

int Tok::intVal() { return static_cast<int>(std::atoll(next().c_str())); }

DigitSet parseDigits(const std::string& text) {
  DigitSet ds;
  Tok t(text);
  if (!t.has()) {
    return ds;
  }
  const std::string head = t.next();
  if (head != "DIGITS") {
    return ds;
  }
  ds.size = t.intVal();
  const int count = t.intVal();
  for (int i = 0; i < count; i++) {
    const std::string tag = t.next();
    if (tag != "SAMPLE") {
      break;
    }
    const int label = t.intVal();
    const int pred = t.intVal();
    const std::string name = t.next();
    const int n = t.intVal();
    std::vector<double> px = zeros(n);
    for (int j = 0; j < n; j++) {
      px[j] = t.num();
    }
    ds.items.push_back(Digit(label, pred, std::move(px), name));
  }
  return ds;
}

Weights parseWeights(const std::string& text) {
  Weights w;
  Tok t(text);
  if (!t.has()) {
    return w;
  }
  const std::string head = t.next();
  if (head != "MODEL") {
    return w;
  }
  t.intVal();
  while (t.has()) {
    const std::string tag = t.next();
    if (tag != "W") {
      break;
    }
    const std::string name = t.next();
    const int n = t.intVal();
    std::vector<double> data = zeros(n);
    for (int i = 0; i < n; i++) {
      data[i] = t.num();
    }
    if (name == "w1") {
      w.w1 = std::move(data);
    } else if (name == "b1") {
      w.b1 = std::move(data);
    } else if (name == "w2") {
      w.w2 = std::move(data);
    } else if (name == "b2") {
      w.b2 = std::move(data);
    } else if (name == "w3") {
      w.w3 = std::move(data);
    } else if (name == "b3") {
      w.b3 = std::move(data);
    }
  }
  return w;
}

std::string sampleName(int i) { return "样本" + std::to_string(i + 1); }

}  // namespace core
