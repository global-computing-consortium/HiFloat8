import numpy as np


def To_HiF8(x):
    # Only support round half to away for Matrix
    x = np.array(x)

    M = x.shape[0]
    N = x.shape[1]

    res = np.zeros((M,N), dtype=np.float32)
    for i in range(M):
        for j in range(N): 
            z = x[i,j]
            if z == np.nan or z == np.inf or z == -np.inf:
                res_z = z
            else: 
                s = 1.0 if z >= 0 else -1.0
                tmp = np.abs(z)               
                if tmp >= 2.0**15*1.25:
                    res_z = s * np.inf
                elif tmp < 2.0**(-23): 
                    res_z = 0.0
                else: 
                    E = np.floor(np.log2(tmp))
                    E = -22 if E == -23 else E
                    absE = np.abs(E)
                    if absE <= 3:       # 3-bit Mantissa
                        res_z = np.floor(tmp*2.0**(-E + 3) + 0.5)*2.0**(E - 3) * s
                    elif absE <= 7:     # 2-bit Mantissa
                        res_z = np.floor(tmp*2.0**(-E + 2) + 0.5)*2.0**(E - 2) * s
                    elif absE <= 15:    # 1-bit Mantissa
                        res_z = np.floor(tmp*2.0**(-E + 1) + 0.5)*2.0**(E - 1) * s
                    else:               # 0-bit Mantissa
                        res_z = np.floor(tmp*2.0**(-E) + 0.5)*2.0**E * s
            res[i,j] = res_z
    return res

def To_FP8(x, fmt = 'E4M3'):
    # Convert data to FP8
    # fmt: 'E4M3' for default, others for 'E5M2'
    # Round Mode: Round Half tie to Even; if overflow, clip to upper bound
    x = np.float64(x)
    S = np.ones_like(x)
    S[x < 0] = -1
    tmp = np.abs(x)
    E = np.floor(np.log2(tmp + 2**(-1000)))

    if fmt == 'E4M3':
        E[E < -6] = -6
        res_abs = np.round(tmp*2**(-E + 3))*2**(E - 3)
        res_abs[np.abs(res_abs) > 448] = 448    # Clip to upper-bound
        res = S * res_abs
    else:   # for 'E5M2'
        E[E < -14] = -14
        res_abs = np.round(tmp*2**(-E + 2))*2**(E - 2)
        res_abs[np.abs(res_abs) > 57344] = 57344    # Clip to upper-bound
        res = S * res_abs

    return res

def To_MXFP8(x, fmt = 'E4M3'):
    # Convert data to MXFP8, Column-wise
    # fmt: 'E4M3' for default, others for 'E5M2'
    # Round Mode: Round Half tie to Even; if overflow, clip to upper bound
    x = np.float64(x)
    G = 32
    M = x.shape[0]
    N = x.shape[1]

    Mcnt = np.ceil(M/G).astype(int)

    res = np.zeros((M,N))
    E8 = np.zeros((Mcnt,N))
    grp = np.zeros((M,N))

    if fmt == 'E4M3':
        Emax = 8    # Max Exp of E4M3
    else:
        Emax = 15   # Max Exp of E5M2

    for i in range(Mcnt):
        for j in range(N):
            ori = x[i*G:i*G+G,j]
            tmp = np.abs(ori)   # abs of grp values

            E = np.floor(np.log2(tmp + 2**(-1000)))          
            E8g = np.max(E) - Emax
            E8g = -127 if (E8g < -127) else E8g

            xgi = ori * 2 ** (-E8g)        # in_grp convert input
            xgo = To_FP8(xgi,fmt = fmt)    # in_grp convert output
            xgr = xgo * 2 ** (E8g)         # restored grp values

            E8[i,j] = E8g
            grp[i*G:i*G+G,j] = xgo
            res[i*G:i*G+G,j] = xgr
    
    return res, E8, grp

# M = 32
# N = 6
# mu = 0
# sigma = 0.8
# x = np.random.normal(mu, sigma, (M, N))
# print(x,'\n')
# fmt = 'E4M3'
# # fmt = 'E5M2

# f1 = To_FP8(x,fmt)
# print(f1,'\n')

# f2, E8, grp = To_MXFP8(x,fmt)
# print(f2,'\n')
# print(E8,'\n')
# print(grp,'\n')

# d = f2 - f1
# print(d,'\n')