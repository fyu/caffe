#include <vector>

#include "caffe/layers/check_label_layer.hpp"

namespace caffe {

template <typename Dtype>
void CheckLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.check_label_param().top_k();

  has_ignore_label_ =
      this->layer_param_.check_label_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.check_label_param().ignore_label();
  }
  ignore_correct_label_ =
      this->layer_param_.check_label_param().ignore_correct_label();
}

template <typename Dtype>
void CheckLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  label_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.check_label_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
    << "Number of labels must match number of predictions; "
    << "label count (number of labels) must be N*H*W, "
    << "with integer values in {0, 1, ..., C-1}.";
  top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void CheckLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_label = top[0]->mutable_cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      Dtype *correct_label = top_label + i * inner_num_ + j;
      if (has_ignore_label_ && label_value == ignore_label_) {
        *correct_label = ignore_correct_label_;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      *correct_label = 0;
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          *correct_label = 1;
          break;
        }
      }
    }
  }
}

template <typename Dtype>
void CheckLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(ConcatLayer);
#endif

INSTANTIATE_CLASS(CheckLabelLayer);
REGISTER_LAYER_CLASS(CheckLabel);

}  // namespace caffe
