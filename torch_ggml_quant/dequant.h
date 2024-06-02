#pragma once

#include <torch/extension.h>

void dequantize_fp32_cpu(
    torch::Tensor data,
    torch::Tensor out,
    int quant_type
);

void dequantize_fp32_cuda(
    torch::Tensor data,
    torch::Tensor out,
    int quant_type
);

void dequantize_fp16_cuda(
    torch::Tensor data,
    torch::Tensor out,
    int quant_type
);
