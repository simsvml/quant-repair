#pragma once

// Copied from ggml-quants.h and other llama.cpp files (as of 2024-05-20,
// revision 917dc8cf).  Used under the terms of LICENSE_GGML.

#include "quant_formats.h"

// Dequantization
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);
void dequantize_row_q4_1(const block_q4_1 * x, float * y, int64_t k);
void dequantize_row_q5_0(const block_q5_0 * x, float * y, int64_t k);
void dequantize_row_q5_1(const block_q5_1 * x, float * y, int64_t k);
void dequantize_row_q8_0(const block_q8_0 * x, float * y, int64_t k);
//void dequantize_row_q8_1(const block_q8_1 * x, float * y, int64_t k);

void dequantize_row_q2_K(const block_q2_K * x, float * y, int64_t k);
void dequantize_row_q3_K(const block_q3_K * x, float * y, int64_t k);
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k);
void dequantize_row_q5_K(const block_q5_K * x, float * y, int64_t k);
void dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k);
void dequantize_row_q8_K(const block_q8_K * x, float * y, int64_t k);
