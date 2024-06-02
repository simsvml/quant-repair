#pragma once

// Copied from ggml-quants.h and other llama.cpp files (as of 2024-06-02,
// revision e141ce62).  Used under the terms of LICENSE_GGML.

#include "quant_formats.h"

void quantize_row_q8_0(const float * x, void * y, int64_t k);
