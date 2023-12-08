#include <torch/extension.h>

torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);

void ops_convert_s4_k_m8(
  torch::Tensor _weight_dest,
  torch::Tensor _quant_scales_zeros_dest,
  torch::Tensor _workspace,
  torch::Tensor _quant_weight_src,
  torch::Tensor _quant_scales,
  torch::Tensor _quant_zeros,
  int m,
  int k,
  int group_size);

torch::Tensor int4_f16_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scales_zeros);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
  m.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");
  m.def("ops_convert_s4_k_m8", &ops_convert_s4_k_m8, "convert kernel.");
  m.def("int4_f16_gemm", &int4_f16_gemm, "weight int4 activation float16 gemm kernel.");
}
