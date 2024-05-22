// Copied from ggml-cuda/convert.cu and other llama.cpp files (as of 2024-05-20,
// revision 917dc8cf).  Used under the terms of LICENSE_GGML.

// PyTorch adds `-D` flags to disable fp16 conversions and operators when
// building extensions.  We want those conversions to be present because the
// copied GGML code uses them.  We don't include any PyTorch headers here.
#ifdef __CUDA_NO_HALF_CONVERSIONS__
# undef __CUDA_NO_HALF_CONVERSIONS__
#endif
//#ifdef __CUDA_NO_HALF_OPERATORS__
//# undef __CUDA_NO_HALF_OPERATORS__
//#endif
//#ifdef __CUDA_NO_HALF2_OPERATORS__
//# undef __CUDA_NO_HALF2_OPERATORS__
//#endif

#include "quant_formats.h"

#include <cuda_fp16.h>


//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#else
    const int64_t is = tid/16;  // 0 or 1
    const int64_t il = tid%16;  // 0...15
    const uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;
    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

#if QK_K == 256
    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#else
    const int64_t tid = threadIdx.x;
    const int64_t is  = tid/16;  // 0 or 1
    const int64_t il  = tid%16;  // 0...15
    const int64_t im  = il/8;    // 0...1
    const int64_t in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    const uint8_t q = x[i].qs[il] >> (2*is);
    const uint8_t h = x[i].hmask[in] >> (2*is + im);
    const float   d = (float)x[i].d;

    if (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } else {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

#if QK_K == 256
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIdx.x;

#if QK_K == 256
    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
#else
    const int64_t tid = threadIdx.x;
    const uint8_t * q = x[i].qs;
    dst_t * y = yy + i*QK_K;
    const float d = (float)x[i].dm[0];
    const float m = (float)x[i].dm[1];
    y[tid+ 0] = d * (x[i].scales[0] & 0xF) * (q[tid] & 0xF) - m * (x[i].scales[0] >> 4);
    y[tid+32] = d * (x[i].scales[1] & 0xF) * (q[tid] >>  4) - m * (x[i].scales[1] >> 4);
#endif
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIdx.x;

#if QK_K == 256
    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#else
    const int64_t tid = threadIdx.x;
    const uint8_t q = x[i].qs[tid];
    const int64_t im = tid/8;  // 0...3
    const int64_t in = tid%8;  // 0...7
    const int64_t is = tid/16; // 0 or 1
    const uint8_t h = x[i].qh[in] >> im;
    const float d = x[i].d;
    dst_t * y = yy + i*QK_K + tid;
    y[ 0] = d * x[i].scales[is+0] * ((q & 0xF) - ((h >> 0) & 1 ? 0 : 16));
    y[32] = d * x[i].scales[is+2] * ((q >>  4) - ((h >> 4) & 1 ? 0 : 16));
#endif
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;
#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#else

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/16;         // 0 or 1
    const int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    const float d = x[i].d;

    const uint8_t   ql = x[i].ql[16*ip + il];
    const uint8_t   qh = x[i].qh[il] >> (2*ip);
    const int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}


template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q2_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q3_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q5_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q6_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}


void dequantize_row_q2_K_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q2_K_cuda(vx, (ggml_half*)y, k, stream);
}

void dequantize_row_q3_K_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q3_K_cuda(vx, (ggml_half*)y, k, stream);
}

void dequantize_row_q4_K_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q4_K_cuda(vx, (ggml_half*)y, k, stream);
}

void dequantize_row_q5_K_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q5_K_cuda(vx, (ggml_half*)y, k, stream);
}

void dequantize_row_q6_K_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q6_K_cuda(vx, (ggml_half*)y, k, stream);
}


void dequantize_row_q2_K_cuda_fp32(const void * vx, float * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q2_K_cuda(vx, y, k, stream);
}

void dequantize_row_q3_K_cuda_fp32(const void * vx, float * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q3_K_cuda(vx, y, k, stream);
}

void dequantize_row_q4_K_cuda_fp32(const void * vx, float * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q4_K_cuda(vx, y, k, stream);
}

void dequantize_row_q5_K_cuda_fp32(const void * vx, float * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q5_K_cuda(vx, y, k, stream);
}

void dequantize_row_q6_K_cuda_fp32(const void * vx, float * y, const int64_t k, cudaStream_t stream) {
    dequantize_row_q6_K_cuda(vx, y, k, stream);
}
