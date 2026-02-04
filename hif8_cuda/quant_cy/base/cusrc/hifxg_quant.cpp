#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void hif8_quant_cuda(const torch::Tensor x, torch::Tensor result);
void hif8_quant_cuda_fp16(const torch::Tensor x, torch::Tensor result);
void hif8_quant_cuda_bf16(const torch::Tensor x, torch::Tensor result);


void hif8_quant(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    hif8_quant_cuda(x, result);
}

void hif8_quant_fp16(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    hif8_quant_cuda_fp16(x, result);
}

void hif8_quant_bf16(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    hif8_quant_cuda_bf16(x, result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("hif8_quant", &hif8_quant, "hif8_quant", py::arg("x"), py::arg("result"));
    m.def("hif8_quant_fp16", &hif8_quant_fp16, "hif8_quant_fp16", py::arg("x"), py::arg("result"));
    m.def("hif8_quant_bf16", &hif8_quant_bf16, "hif8_quant_bf16", py::arg("x"), py::arg("result"));
}

