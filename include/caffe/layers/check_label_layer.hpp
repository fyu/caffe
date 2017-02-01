#ifndef CAFFE_CHECK_LABEL_LAYER_HPP_
#define CAFFE_CHECK_LABEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CheckLabelLayer : public Layer<Dtype> {
public:
  explicit CheckLabelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CheckLabel"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int label_axis_;
  int inner_num_;
  int outer_num_;

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;

  int ignore_correct_label_;

  int top_k_;
};

}  // namespace caffe

#endif  // CAFFE_CHECK_LABEL_LAYER_HPP_
