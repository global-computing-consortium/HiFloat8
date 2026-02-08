# HiFloat8

HiFloat8: High-Performance HiFloat8 Quantization Library.

HiFloat8 is a library designed for efficient Float8 quantization and simulation across different hardware backends, including NVIDIA CUDA and Huawei Ascend NPU. It provides high-performance kernels for pseudo-quantization, enabling researchers to simulate HiFloat8 precision in deep learning models.

Installation & Verification
1. CUDA Version (NVIDIA GPUs)
To build and verify the CUDA-accelerated operators, follow these steps:

```bash
# Build the CUDA kernels
bash build.sh

# Run the verification script
python hif8_bf16.py

# Verification:
If the output: displays ABS diff max (zero values): 0
The installation is successful, and the results match the reference.
```
2. NPU Version (Huawei Ascend)
To build and verify the NPU-accelerated operators, follow these steps:
```bash
# Build the NPU kernels
bash build_npu_ops.sh

# Run the verification script
python hif8_bf16.py

# Verification:
If the output: displays ABS diff max (zero values): 0.
The installation is working correctly on the Ascend hardware.
```

3. Usage: Standard Linear Layer Simulation (GPU Example). To simulate a HiFloat8 Linear layer using pseudo-quantization on a GPU, you should quantize both the input $x$ and the weights $w$ before performing the standard linear operation. The standard workflow for the GPU platform is as follows:
```bash
import torch
from quant_cy import QType, quant_dequant_float #gpu

# 1. Prepare your input and weights
# 2. Apply quant-dequant simulation
# Note: Ensure tensors are on the correct device (e.g., .cuda())
qtype_str = 'hif8'
print('Qtype string: %s '%(qtype_str))
quant_type = QType(qtype_str).dim(0) 
x_sim = quant_dequant_float(x.cuda(), quant_type, force_py=False, force_fp32=True)
w_sim = quant_dequant_float(w.cuda(), quant_type, force_py=False, force_fp32=True)

# 3. Execute the linear layer
y = torch.nn.functional.linear(x_sim, w_sim)
```

4. Usage: Standard Linear Layer Simulation (NPU Example):
```bash
import torch
from quant_cy_npu import QType, quant_dequant_float #npu

# 1. Prepare your input and weights
# 2. Apply quant-dequant simulation
# Note: Ensure tensors are on the correct device (e.g., .npu())
qtype_str = 'hif8'
print('Qtype string: %s '%(qtype_str))
quant_type = QType(qtype_str).dim(0) 
x_sim = quant_dequant_float(x.npu(), quant_type, force_py=False, force_fp32=True)
w_sim = quant_dequant_float(w.npu(), quant_type, force_py=False, force_fp32=True)

# 3. Execute the linear layer
y = torch.nn.functional.linear(x_sim, w_sim)
```

# Citation
If you find this work useful for your research, please cite the following paper:
```bash
@misc{luo2024ascendhifloat8formatdeep,
      title={Ascend HiFloat8 Format for Deep Learning}, 
      author={Yuanyong Luo and Zhongxing Zhang and Richard Wu and Hu Liu and Ying Jin and Kai Zheng and Minmin Wang and Zhanying He and Guipeng Hu and Luyao Chen and Tianchi Hu and Junsong Wang and Minqi Chen and Mikhaylov Dmitry and Korviakov Vladimir and Bobrin Maxim and Yuhao Hu and Guanfu Chen and Zeyi Huang},
      year={2024},
      eprint={2409.16626},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.16626}, 
}
```

