#include "torch/extension.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"



void run_hif8_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len);
void run_hif8_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len);
void run_mxfp8e4m3_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_mxfp8e4m3_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_nvf4_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_nvf4_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_hifx_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit);
void run_hifx_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit);
void run_hifxsub_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit);
void run_hifxsub_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit);



void hif8_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int len = x.numel();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hif8_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), len);
}

void hif8_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int len = x.numel();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hif8_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), len);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hif8_quant", &hif8_quant, "hif8_quant");
    m.def("hif8_quant_bf16", &hif8_quant_bf16, "hif8_quant_bf16");
}
