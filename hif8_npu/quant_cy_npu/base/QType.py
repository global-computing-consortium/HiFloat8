import re
from copy import deepcopy


class QType():
    # declare datatype and default values 
    desc: str
    exp_bits: int = -1
    man_bits: int = -1
    k_bits: int = -1
    k_outer_bits: int = 0
    blk_size: int = 1
    blk_outer_size: int = 1
    exp_max: int = -1
    exp_min: int = -1
    k_max: int = -1
    fp_val_max: float = -1
    q_dim: int = -1
    man_shift_bit: int = -1
    exp_offset: int = 0
    do_carry: bool = True

    def __init__(self, desc: str):
        # some special ones 
        self.desc = desc 
        if desc in ['fp16', 'fp32', 'bf16', 'hif8']:
            ... 

    def dim_(self, dim: int):
        # inplace function 
        self.q_dim = dim 
        return self 

    def dim(self, dim: int):
        out = deepcopy(self)
        out.q_dim = dim 
        return out 

    def copy(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        return f'QType: {self.desc}   Dim: {self.q_dim}  ExpOffset: {self.exp_offset}'


if __name__=='__main__':
    from copy import deepcopy 
    t = QType('e2m1k8b8')
    t2 = deepcopy(t)
    print()
