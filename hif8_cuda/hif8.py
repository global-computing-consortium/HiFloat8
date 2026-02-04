import torch 
from quant_cy import QType, quant_dequant_float
import numpy as np 
import toF8

np.random.seed(42)

N = 512
M = 512
NN = N
MM = M 
# x = np.load('problem.npy')[...,None]
x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N))).astype(np.float32) * 0.5
x_torch = torch.from_numpy(x)
print(x.shape)


qtype_str = 'hif8'
print('Qtype string: %s '%(qtype_str))
quant_type = QType(qtype_str).dim(0)

y0 = toF8.To_HiF8(x)

# y1 = quant_dequant_float(x_torch, quant_type, force_py=True, force_fp32=True).cpu().numpy()

y2 = quant_dequant_float(x_torch.cuda(), quant_type, force_py=False, force_fp32=True).cpu().numpy()

# diff = np.abs(y0 - y1)
# print('ABS diff max (numpy <-> torch ):', np.max(diff))
diff = np.abs(y0 - y2)
print('ABS diff max (numpy <-> kernel):', np.max(diff))
# diff = np.abs(y1 - y2)
# print('ABS diff max (torch <-> kernel):', np.max(diff))
