#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include "quant_formats.h"

void dequantize_row_q2_K_cuda_fp16(
    const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_q3_K_cuda_fp16(
    const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_q4_K_cuda_fp16(
    const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_q5_K_cuda_fp16(
    const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_q6_K_cuda_fp16(
    const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);

void dequantize_row_q2_K_cuda_fp32(
    const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_q3_K_cuda_fp32(
    const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_q4_K_cuda_fp32(
    const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_q5_K_cuda_fp32(
    const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_q6_K_cuda_fp32(
    const void * vx, float * y, const int64_t k, cudaStream_t stream);


void dequantize_row_iq2_xxs_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq2_xs_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq2_s_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq3_xxs_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq3_s_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq1_s_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq4_nl_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq1_m_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq4_xs_cuda_fp16(
        const void * vx, ggml_half_placeholder * y, const int64_t k, cudaStream_t stream);

void dequantize_row_iq2_xxs_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq2_xs_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq2_s_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq3_xxs_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq3_s_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq1_s_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq4_nl_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq1_m_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
void dequantize_row_iq4_xs_cuda_fp32(
        const void * vx, float * y, const int64_t k, cudaStream_t stream);
