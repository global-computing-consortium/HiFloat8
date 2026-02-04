#include "tensorutils.h"




template <typename T>
class Hif8Vec{
public:
    aifunc Hif8Vec(){}
    aifunc void Init(GM_ADDR src, GM_ADDR dst, int len){
        int n_percore = CeilDiv(CeilDiv(len, get_block_num()), BATCH) * BATCH;
        n1 = n_percore*get_block_idx();
        n2 = n1 + n_percore;
        if (n2 > len){ n2 = len; }

        x = Tensor<T, PGM>(src);
        out = Tensor<T, PGM>(dst);

        int offset = 0;
        xbuf = DBuff<T, PUB>(0, BATCH, offset);
        outbuf = DBuff<T, PUB>(0, BATCH, offset);
        expbuf = Tensor<float, PUB>(0, BATCH, offset);
        expabsbuf = Tensor<float, PUB>(0, BATCH, offset);

        le15buf = Tensor<float, PUB>(0, 64, offset);
        le7buf = Tensor<float, PUB>(0, 64, offset);
        le3buf = Tensor<float, PUB>(0, 64, offset);
        eq0buf = Tensor<float, PUB>(0, 64, offset);
        posinfbuf = Tensor<float, PUB>(0, 64, offset);
        neginfbuf = Tensor<float, PUB>(0, 64, offset);
        twosbuf = Tensor<float, PUB>(0, 64, offset);
        onesbuf = Tensor<float, PUB>(0, 64, offset);
        zerosbuf = Tensor<float, PUB>(0, 64, offset);
        expbias1 = Tensor<float, PUB>(0, BATCH, offset);
        expbias2 = Tensor<float, PUB>(0, BATCH, offset);
        expbias3 = Tensor<float, PUB>(0, BATCH, offset);

        xf32buf = Tensor<float, PUB>(0, BATCH, offset);
        outf32buf = Tensor<float, PUB>(0, BATCH, offset);

        expmask = Tensor<uint32_t, PUB>(0, 64, offset);
        posinfval = Tensor<uint32_t, PUB>(0, 64, offset);
        neginfval = Tensor<uint32_t, PUB>(0, 64, offset);
        vector_dup(expmask.ptr(), 0x7F800000, 1, 1, 1, 8, 8);
        vector_dup(posinfval.ptr(), 0x7F800000, 1, 1, 1, 8, 8);
        vector_dup(neginfval.ptr(), 0xFF800000, 1, 1, 1, 8, 8);
        vector_dup(twosbuf.ptr(), 0.5f, 1, 1, 1, 8, 8);
        vector_dup(onesbuf.ptr(), 1.0f, 1, 1, 1, 8, 8);
        vector_dup(zerosbuf.ptr(), 0.0f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_ALL);
    }

    aifunc void Compute(){
        in_empty.setall();
        out_empty.setall();

        int cnt = 0;
        for (int n=n1; n<n2; n+=BATCH){
            Compute(n, cnt);
        }

        in_empty.release();
        out_empty.release();
    }

    aifunc void Compute(int &n, int &cnt){
        Tensor<float, PUB> inp_tsr;
        Tensor<float, PUB> out_tsr;

        in_empty.wait();
        copy_gm_to_ubuf(xbuf.get(cnt).vptr(), x[n].vptr(), 0, 1, BATCH*sizeof(T)/32, 0, 0);
        in_ready.set();

        in_ready.wait();
        out_empty.wait();
        // do type conversion if necessary
        if constexpr(std::is_same<T, float>::value){
            inp_tsr = xbuf.get(cnt);
            out_tsr = outbuf.get(cnt);
        }else{
            inp_tsr = xf32buf;
            out_tsr = outf32buf;
            vconv_bf162f32(inp_tsr.ptr(), xbuf.get(cnt).ptr(), BATCH/64, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }


        // ---- get exp ---- 
        vand((__ubuf__ uint16_t*)expbuf.ptr(), (__ubuf__ uint16_t*)inp_tsr.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), BATCH/64, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);
        

        // ---- get abs exp ----
        // vrec(expabsbuf.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 8, 8); // fuck. this cannot produce correct recip
        vnot((__ubuf__ uint16_t*) expabsbuf.ptr(), (__ubuf__ uint16_t*) expbuf.ptr(), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vand((__ubuf__ uint16_t*)expabsbuf.ptr(), (__ubuf__ uint16_t*)expabsbuf.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), BATCH/64, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);
        vadds((__ubuf__ int32_t*)expabsbuf.ptr(), (__ubuf__ int32_t*)expabsbuf.ptr(), -8388608, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmax(expabsbuf.ptr(), expbuf.ptr(), expabsbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // ---- adjust the exp according to absexp ----
        // compare with different thresholds 
        vcmpvs_le((__ubuf__ uint8_t*)le15buf.ptr(), expabsbuf.ptr(), 32768.0f, BATCH/64, 1, 1, 8, 8);
        vcmpvs_le((__ubuf__ uint8_t*)le7buf.ptr(), expabsbuf.ptr(), 128.0f, BATCH/64, 1, 1, 8, 8);
        vcmpvs_le((__ubuf__ uint8_t*)le3buf.ptr(), expabsbuf.ptr(), 8.0f, BATCH/64, 1, 1, 8, 8);
        vcmpvs_lt((__ubuf__ uint8_t*)posinfbuf.ptr(), inp_tsr.ptr(), 40960.0f, BATCH/64, 1, 1, 8, 8);  
        vcmpvs_gt((__ubuf__ uint8_t*)neginfbuf.ptr(), inp_tsr.ptr(), -40960.0f, BATCH/64, 1, 1, 8, 8);  
        uint32_t zero_thresh_uint = 0x34000000;
        float zero_thresh = *(float*) &zero_thresh_uint;
        vcmpvs_ge((__ubuf__ uint8_t*)eq0buf.ptr(), expbuf.ptr(), zero_thresh, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(onesbuf.vptr());
        uint32_t expmin_uint = 0x34800000;  // 2^-22
        float expmin = *(float*) &expmin_uint;
        vmaxs(expbuf.ptr(), expbuf.ptr(), expmin, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vsel(expbias1.ptr(), twosbuf.ptr(), le15buf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        vsel(expbias2.ptr(), twosbuf.ptr(), le7buf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        vsel(expbias3.ptr(), twosbuf.ptr(), le3buf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        pipe_barrier(PIPE_V);
        // manipulate exp 
        vmul(expbias1.ptr(), expbias2.ptr(), expbias1.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        vmul(expbuf.ptr(), expbias3.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(expbuf.ptr(), expbias1.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // x div exp and round and mul exp 
        vdiv(out_tsr.ptr(), inp_tsr.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32a(out_tsr.ptr(), out_tsr.ptr(), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(zerosbuf.vptr());
        pipe_barrier(PIPE_V);
        vsel(out_tsr.ptr(), out_tsr.ptr(), eq0buf.vptr(), BATCH/64, 1, 1, 1, 8, 8, 1, 1);
        pipe_barrier(PIPE_V);
        // final output 
        vmul(out_tsr.ptr(), out_tsr.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // fill posinf 
        set_cmpmask(posinfval.vptr());
        pipe_barrier(PIPE_V);
        vsel(out_tsr.ptr(), out_tsr.ptr(), posinfbuf.vptr(), BATCH/64, 1, 1, 1, 8, 8, 1, 1);
        pipe_barrier(PIPE_V);
        // fill neginf 
        set_cmpmask(neginfval.vptr());
        pipe_barrier(PIPE_V);
        vsel(out_tsr.ptr(), out_tsr.ptr(), neginfbuf.vptr(), BATCH/64, 1, 1, 1, 8, 8, 1, 1);
        pipe_barrier(PIPE_V);

        // do type conversion if necessary 
        if constexpr(std::is_same<T, float>::value){
        }else{
            vconv_f322bf16a(outbuf.get(cnt).ptr(), out_tsr.ptr(), BATCH/64, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
        }

        out_ready.set();
        in_empty.set();

        out_ready.wait();
        copy_ubuf_to_gm(out[n].vptr(), outbuf.get(cnt).vptr(), 0, 1, BATCH*sizeof(T)/32, 0, 0);
        out_empty.set();

        cnt++;
    }

private:
    TPipe pipe;
    int n1, n2;
    Tensor<T, PGM> x, out;
    DBuff<T, PUB> xbuf, outbuf; 
    Tensor<float, PUB> xf32buf, outf32buf;
    Tensor<float, PUB> expbuf, expabsbuf, le15buf, le7buf, le3buf, eq0buf, posinfbuf, neginfbuf, twosbuf, onesbuf, zerosbuf, expbias1, expbias2, expbias3;
    Tensor<uint32_t, PUB> expmask, posinfval, neginfval;

    DEvent<PIPE_MTE2, PIPE_V> in_ready{3,4};
    DEvent<PIPE_V, PIPE_MTE2> in_empty{3,4};
    DEvent<PIPE_V, PIPE_MTE3> out_ready{3,4};
    DEvent<PIPE_MTE3, PIPE_V> out_empty{3,4};

    static constexpr int BATCH = 512;
};


extern "C" __global__ __aicore__ void hif8_kernel(GM_ADDR xmtx, GM_ADDR out, int len){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        Hif8Vec<float> vec;
        vec.Init(xmtx, out, len);
        vec.Compute();
#endif 
    }
}

extern "C" __global__ __aicore__ void hif8_kernel_bf16(GM_ADDR xmtx, GM_ADDR out, int len){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        Hif8Vec<bfloat16_t> vec;
        vec.Init(xmtx, out, len);
        vec.Compute();
#endif 
    }
}

void run_hif8_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len) {
    hif8_kernel<<<40, nullptr, stream>>>(xmtx, out, len);
}

void run_hif8_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len) {
    hif8_kernel_bf16<<<40, nullptr, stream>>>(xmtx, out, len);
}
