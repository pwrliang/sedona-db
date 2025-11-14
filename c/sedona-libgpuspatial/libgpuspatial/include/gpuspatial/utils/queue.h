#pragma once

#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/queue_view.h"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_scalar.hpp"
#include "rmm/device_uvector.hpp"

namespace gpuspatial {

template <typename T, typename SIZE_T = uint32_t>
class Queue {
 public:
  using value_type = T;
  using device_t = QueueView<T, SIZE_T>;

  Queue() {}

  void Init(const rmm::cuda_stream_view& stream, SIZE_T capacity) {
    if (data_ == nullptr) {
      data_ = std::make_unique<rmm::device_uvector<T>>(capacity, stream);
    } else {
      data_->resize(capacity, stream);
    }
    if (counter_ == nullptr) {
      counter_ = std::make_unique<rmm::device_scalar<SIZE_T>>(stream);
    }
  }

  void Clear(const rmm::cuda_stream_view& stream) {
    counter_->set_value_to_zero_async(stream);
  }

  void set_size(const rmm::cuda_stream_view& stream, SIZE_T n) {
    counter_->set_value_async(n, stream);
  }

  SIZE_T size(const rmm::cuda_stream_view& stream) const {
    return counter_->value(stream);
  }

  T* data() { return data_->data(); }

  const T* data() const { return data_->data(); }

  device_t DeviceObject() {
    return device_t(ArrayView<T>(data_->data(), capacity()), counter_->data());
  }

  void Swap(Queue<T>& rhs) {
    data_.swap(rhs.data_);
    counter_.Swap(rhs.counter_);
  }

  void shrink_to_fit(const rmm::cuda_stream_view& stream) {
    auto s = size(stream);
    data_->resize(s, stream);
    data_->shrink_to_fit(stream);
  }

  size_t capacity() const { return data_->capacity(); }

 private:
  std::unique_ptr<rmm::device_uvector<T>> data_;
  std::unique_ptr<rmm::device_scalar<SIZE_T>> counter_;
};

}  // namespace gpuspatial
