#pragma once

#include <torch/extension.h>

void quantize_fp32_cpu(
    torch::Tensor data,
    torch::Tensor out,
    int quant_type
);
