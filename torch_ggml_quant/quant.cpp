#include "quant.h"
#include <torch/extension.h>

#include <iostream>

#include "quant_cpu.h"

template <typename block_t, unsigned QUANT_BLOCK_SIZE, typename quantize_func_t>
void quantize_fp32_cpu_impl(
    torch::Tensor data,
    torch::Tensor out,
    quantize_func_t quantize_func
) {
    TORCH_CHECK(out.device().is_cpu(), "out must be a cpu tensor");
    TORCH_CHECK(out.scalar_type() == torch::kByte, "out must be a byte tensor");
    auto out_size = out.sizes();
    TORCH_CHECK(out_size.size() == 2, "out must have 2 dimensions");
    TORCH_CHECK(out_size[1] == sizeof(block_t), "out.shape[1] must match block size");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    TORCH_CHECK(data.device().is_cpu(), "data must be a cpu tensor");
    TORCH_CHECK(data.scalar_type() == torch::kFloat32, "data must be a float32 tensor");
    auto data_size = data.sizes();
    TORCH_CHECK(data_size.size() == 2, "data must have 2 dimensions");
    TORCH_CHECK(data_size[0] == data_size[0], "data.shape[0] must match data.shape[0]");
    TORCH_CHECK(data_size[1] == QUANT_BLOCK_SIZE,
        "data.shape[1] must match quantized block size");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");

    void* out_ptr = out.mutable_data_ptr();
    TORCH_CHECK((uintptr_t)out_ptr % alignof(block_t) == 0,
        "out ptr is misaligned for block type");
    block_t* out_block_ptr = (block_t*)out_ptr;

    const float* data_ptr = data.const_data_ptr<float>();

    quantize_func(data_ptr, (void*)out_block_ptr, data_size[0] * QUANT_BLOCK_SIZE);
}

void quantize_fp32_cpu(
    torch::Tensor data,
    torch::Tensor out,
    int quant_type
) {
    switch (quant_type) {
        /*
        case GGML_TYPE_Q4_0:
            quantize_fp32_cpu_impl<block_q4_0, QK4_0>(data, out, quantize_row_q4_0);
            break;
        case GGML_TYPE_Q4_1:
            quantize_fp32_cpu_impl<block_q4_1, QK4_1>(data, out, quantize_row_q4_1);
            break;
        case GGML_TYPE_Q5_0:
            quantize_fp32_cpu_impl<block_q5_0, QK5_0>(data, out, quantize_row_q5_0);
            break;
        case GGML_TYPE_Q5_1:
            quantize_fp32_cpu_impl<block_q5_1, QK5_1>(data, out, quantize_row_q5_1);
            break;
        */
        case GGML_TYPE_Q8_0:
            quantize_fp32_cpu_impl<block_q8_0, QK8_0>(data, out, quantize_row_q8_0);
            break;

        /*
        case GGML_TYPE_Q2_K:
            quantize_fp32_cpu_impl<block_q2_K, QK_K>(data, out, quantize_row_q2_K);
            break;
        case GGML_TYPE_Q3_K:
            quantize_fp32_cpu_impl<block_q3_K, QK_K>(data, out, quantize_row_q3_K);
            break;
        case GGML_TYPE_Q4_K:
            quantize_fp32_cpu_impl<block_q4_K, QK_K>(data, out, quantize_row_q4_K);
            break;
        case GGML_TYPE_Q5_K:
            quantize_fp32_cpu_impl<block_q5_K, QK_K>(data, out, quantize_row_q5_K);
            break;
        case GGML_TYPE_Q6_K:
            quantize_fp32_cpu_impl<block_q6_K, QK_K>(data, out, quantize_row_q6_K);
            break;
        case GGML_TYPE_Q8_K:
            quantize_fp32_cpu_impl<block_q8_K, QK_K>(data, out, quantize_row_q8_K);
            break;
        */

        /*
        case GGML_TYPE_IQ2_XXS:
            quantize_fp32_cpu_impl<block_iq2_xxs, QK_K>(data, out, quantize_row_iq2_xxs);
            break;
        case GGML_TYPE_IQ2_XS:
            quantize_fp32_cpu_impl<block_iq2_xs, QK_K>(data, out, quantize_row_iq2_xs);
            break;
        case GGML_TYPE_IQ2_S:
            quantize_fp32_cpu_impl<block_iq2_s, QK_K>(data, out, quantize_row_iq2_s);
            break;
        case GGML_TYPE_IQ3_XXS:
            quantize_fp32_cpu_impl<block_iq3_xxs, QK_K>(data, out, quantize_row_iq3_xxs);
            break;
        case GGML_TYPE_IQ1_S:
            quantize_fp32_cpu_impl<block_iq1_s, QK_K>(data, out, quantize_row_iq1_s);
            break;
        case GGML_TYPE_IQ1_M:
            quantize_fp32_cpu_impl<block_iq1_m, QK_K>(data, out, quantize_row_iq1_m);
            break;
        case GGML_TYPE_IQ4_NL:
            quantize_fp32_cpu_impl<block_iq4_nl, QK_K>(data, out, quantize_row_iq4_nl);
            break;
        case GGML_TYPE_IQ4_XS:
            quantize_fp32_cpu_impl<block_iq4_xs, QK_K>(data, out, quantize_row_iq4_xs);
            break;
        case GGML_TYPE_IQ3_S:
            quantize_fp32_cpu_impl<block_iq3_s, QK_K>(data, out, quantize_row_iq3_s);
            break;
        */

        default:
            TORCH_CHECK(false, "unsupported quant type");
    }
}
