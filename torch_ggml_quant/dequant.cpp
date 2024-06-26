#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>

//#include <ggml-common.h>
//#include <ggml-quants.h>
#include "dequant_cpu.h"
#include "dequant_cuda.h"

template <typename block_t, unsigned DEQUANT_BLOCK_SIZE, typename dequantize_func_t>
void dequantize_fp32_cpu_impl(
    torch::Tensor data,
    torch::Tensor out,
    dequantize_func_t dequantize_func
) {
    TORCH_CHECK(data.device().is_cpu(), "data must be a cpu tensor");
    TORCH_CHECK(data.scalar_type() == torch::kByte, "data must be a byte tensor");
    auto data_size = data.sizes();
    TORCH_CHECK(data_size.size() == 2, "data must have 2 dimensions");
    TORCH_CHECK(data_size[1] == sizeof(block_t), "data.shape[1] must match block size");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");

    TORCH_CHECK(out.device().is_cpu(), "out must be a cpu tensor");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be a float32 tensor");
    auto out_size = out.sizes();
    TORCH_CHECK(out_size.size() == 2, "out must have 2 dimensions");
    TORCH_CHECK(out_size[0] == data_size[0], "out.shape[0] must match data.shape[0]");
    TORCH_CHECK(out_size[1] == DEQUANT_BLOCK_SIZE,
        "out.shape[1] must match dequantized block size");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const void* data_ptr = data.const_data_ptr();
    TORCH_CHECK((uintptr_t)data_ptr % alignof(block_t) == 0,
        "data ptr is misaligned for block type");
    const block_t* data_block_ptr = (const block_t*)data_ptr;

    float* out_ptr = out.mutable_data_ptr<float>();

    dequantize_func(data_block_ptr, out_ptr, data_size[0] * DEQUANT_BLOCK_SIZE);
}

void dequantize_fp32_cpu(
    torch::Tensor data,
    torch::Tensor out,
    int quant_type
) {
    switch (quant_type) {
        case GGML_TYPE_Q4_0:
            dequantize_fp32_cpu_impl<block_q4_0, QK4_0>(data, out, dequantize_row_q4_0);
            break;
        case GGML_TYPE_Q4_1:
            dequantize_fp32_cpu_impl<block_q4_1, QK4_1>(data, out, dequantize_row_q4_1);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_fp32_cpu_impl<block_q5_0, QK5_0>(data, out, dequantize_row_q5_0);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_fp32_cpu_impl<block_q5_1, QK5_1>(data, out, dequantize_row_q5_1);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_fp32_cpu_impl<block_q8_0, QK8_0>(data, out, dequantize_row_q8_0);
            break;

        case GGML_TYPE_Q2_K:
            dequantize_fp32_cpu_impl<block_q2_K, QK_K>(data, out, dequantize_row_q2_K);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_fp32_cpu_impl<block_q3_K, QK_K>(data, out, dequantize_row_q3_K);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_fp32_cpu_impl<block_q4_K, QK_K>(data, out, dequantize_row_q4_K);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_fp32_cpu_impl<block_q5_K, QK_K>(data, out, dequantize_row_q5_K);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_fp32_cpu_impl<block_q6_K, QK_K>(data, out, dequantize_row_q6_K);
            break;
        case GGML_TYPE_Q8_K:
            dequantize_fp32_cpu_impl<block_q8_K, QK_K>(data, out, dequantize_row_q8_K);
            break;

        case GGML_TYPE_IQ2_XXS:
            dequantize_fp32_cpu_impl<block_iq2_xxs, QK_K>(data, out, dequantize_row_iq2_xxs);
            break;
        case GGML_TYPE_IQ2_XS:
            dequantize_fp32_cpu_impl<block_iq2_xs, QK_K>(data, out, dequantize_row_iq2_xs);
            break;
        case GGML_TYPE_IQ2_S:
            dequantize_fp32_cpu_impl<block_iq2_s, QK_K>(data, out, dequantize_row_iq2_s);
            break;
        case GGML_TYPE_IQ3_XXS:
            dequantize_fp32_cpu_impl<block_iq3_xxs, QK_K>(data, out, dequantize_row_iq3_xxs);
            break;
        case GGML_TYPE_IQ1_S:
            dequantize_fp32_cpu_impl<block_iq1_s, QK_K>(data, out, dequantize_row_iq1_s);
            break;
        case GGML_TYPE_IQ1_M:
            dequantize_fp32_cpu_impl<block_iq1_m, QK_K>(data, out, dequantize_row_iq1_m);
            break;
        case GGML_TYPE_IQ4_NL:
            dequantize_fp32_cpu_impl<block_iq4_nl, QK_K>(data, out, dequantize_row_iq4_nl);
            break;
        case GGML_TYPE_IQ4_XS:
            dequantize_fp32_cpu_impl<block_iq4_xs, QK_K>(data, out, dequantize_row_iq4_xs);
            break;
        case GGML_TYPE_IQ3_S:
            dequantize_fp32_cpu_impl<block_iq3_s, QK_K>(data, out, dequantize_row_iq3_s);
            break;

        default:
            TORCH_CHECK(false, "unsupported quant type");
    }
}


template <
    typename block_t,
    unsigned DEQUANT_BLOCK_SIZE,
    typename dst_t,
    typename dequantize_func_t
>
void dequantize_cuda_impl(
    torch::Tensor data,
    torch::Tensor out,
    dequantize_func_t dequantize_func
) {
    TORCH_CHECK(data.device().is_cuda(), "data must be a cuda tensor");
    TORCH_CHECK(data.scalar_type() == torch::kByte, "data must be a byte tensor");
    auto data_size = data.sizes();
    TORCH_CHECK(data_size.size() == 2, "data must have 2 dimensions");
    TORCH_CHECK(data_size[1] == sizeof(block_t), "data.shape[1] must match block size");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");

    TORCH_CHECK(out.device().is_cuda(), "out must be a cuda tensor");
    // out.scalar_type() must be checked by the caller
    auto out_size = out.sizes();
    TORCH_CHECK(out_size.size() == 2, "out must have 2 dimensions");
    TORCH_CHECK(out_size[0] == data_size[0], "out.shape[0] must match data.shape[0]");
    TORCH_CHECK(out_size[1] == DEQUANT_BLOCK_SIZE,
        "out.shape[1] must match dequantized block size");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const void* data_ptr = data.const_data_ptr();
    TORCH_CHECK((uintptr_t)data_ptr % alignof(block_t) == 0,
        "data ptr is misaligned for block type");

    void* out_ptr = out.mutable_data_ptr();
    TORCH_CHECK((uintptr_t)out_ptr % alignof(dst_t) == 0,
        "out ptr is misaligned for output type");
    dst_t* out_ptr_typed = (dst_t*)out_ptr;

    dequantize_func(data_ptr, out_ptr_typed, data_size[0] * DEQUANT_BLOCK_SIZE,
        at::cuda::getCurrentCUDAStream());
}

void dequantize_fp32_cuda(
    torch::Tensor data,
    torch::Tensor out,
    int quant_type
) {
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be a float32 tensor");
    switch (quant_type) {
        case GGML_TYPE_Q2_K:
            dequantize_cuda_impl<block_q2_K, QK_K, float>(
                data, out, dequantize_row_q2_K_cuda_fp32);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_cuda_impl<block_q3_K, QK_K, float>(
                data, out, dequantize_row_q3_K_cuda_fp32);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_cuda_impl<block_q4_K, QK_K, float>(
                data, out, dequantize_row_q4_K_cuda_fp32);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_cuda_impl<block_q5_K, QK_K, float>(
                data, out, dequantize_row_q5_K_cuda_fp32);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_cuda_impl<block_q6_K, QK_K, float>(
                data, out, dequantize_row_q6_K_cuda_fp32);
            break;

        case GGML_TYPE_IQ2_XXS:
            dequantize_cuda_impl<block_iq2_xxs, QK_K, float>(
                data, out, dequantize_row_iq2_xxs_cuda_fp32);
            break;
        case GGML_TYPE_IQ2_XS:
            dequantize_cuda_impl<block_iq2_xs, QK_K, float>(
                data, out, dequantize_row_iq2_xs_cuda_fp32);
            break;
        case GGML_TYPE_IQ2_S:
            dequantize_cuda_impl<block_iq2_s, QK_K, float>(
                data, out, dequantize_row_iq2_s_cuda_fp32);
            break;
        case GGML_TYPE_IQ3_XXS:
            dequantize_cuda_impl<block_iq3_xxs, QK_K, float>(
                data, out, dequantize_row_iq3_xxs_cuda_fp32);
            break;
        case GGML_TYPE_IQ1_S:
            dequantize_cuda_impl<block_iq1_s, QK_K, float>(
                data, out, dequantize_row_iq1_s_cuda_fp32);
            break;
        case GGML_TYPE_IQ1_M:
            dequantize_cuda_impl<block_iq1_m, QK_K, float>(
                data, out, dequantize_row_iq1_m_cuda_fp32);
            break;
        case GGML_TYPE_IQ4_NL:
            dequantize_cuda_impl<block_iq4_nl, QK_K, float>(
                data, out, dequantize_row_iq4_nl_cuda_fp32);
            break;
        case GGML_TYPE_IQ4_XS:
            dequantize_cuda_impl<block_iq4_xs, QK_K, float>(
                data, out, dequantize_row_iq4_xs_cuda_fp32);
            break;
        case GGML_TYPE_IQ3_S:
            dequantize_cuda_impl<block_iq3_s, QK_K, float>(
                data, out, dequantize_row_iq3_s_cuda_fp32);
            break;

        default:
            TORCH_CHECK(false, "unsupported quant type");
    }
}

void dequantize_fp16_cuda(
    torch::Tensor data,
    torch::Tensor out,
    int quant_type
) {
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be a float16 tensor");
    switch (quant_type) {
        case GGML_TYPE_Q2_K:
            dequantize_cuda_impl<block_q2_K, QK_K, ggml_half>(
                data, out, dequantize_row_q2_K_cuda_fp16);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_cuda_impl<block_q3_K, QK_K, ggml_half>(
                data, out, dequantize_row_q3_K_cuda_fp16);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_cuda_impl<block_q4_K, QK_K, ggml_half>(
                data, out, dequantize_row_q4_K_cuda_fp16);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_cuda_impl<block_q5_K, QK_K, ggml_half>(
                data, out, dequantize_row_q5_K_cuda_fp16);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_cuda_impl<block_q6_K, QK_K, ggml_half>(
                data, out, dequantize_row_q6_K_cuda_fp16);
            break;

        case GGML_TYPE_IQ2_XXS:
            dequantize_cuda_impl<block_iq2_xxs, QK_K, ggml_half>(
                data, out, dequantize_row_iq2_xxs_cuda_fp16);
            break;
        case GGML_TYPE_IQ2_XS:
            dequantize_cuda_impl<block_iq2_xs, QK_K, ggml_half>(
                data, out, dequantize_row_iq2_xs_cuda_fp16);
            break;
        case GGML_TYPE_IQ2_S:
            dequantize_cuda_impl<block_iq2_s, QK_K, ggml_half>(
                data, out, dequantize_row_iq2_s_cuda_fp16);
            break;
        case GGML_TYPE_IQ3_XXS:
            dequantize_cuda_impl<block_iq3_xxs, QK_K, ggml_half>(
                data, out, dequantize_row_iq3_xxs_cuda_fp16);
            break;
        case GGML_TYPE_IQ1_S:
            dequantize_cuda_impl<block_iq1_s, QK_K, ggml_half>(
                data, out, dequantize_row_iq1_s_cuda_fp16);
            break;
        case GGML_TYPE_IQ1_M:
            dequantize_cuda_impl<block_iq1_m, QK_K, ggml_half>(
                data, out, dequantize_row_iq1_m_cuda_fp16);
            break;
        case GGML_TYPE_IQ4_NL:
            dequantize_cuda_impl<block_iq4_nl, QK_K, ggml_half>(
                data, out, dequantize_row_iq4_nl_cuda_fp16);
            break;
        case GGML_TYPE_IQ4_XS:
            dequantize_cuda_impl<block_iq4_xs, QK_K, ggml_half>(
                data, out, dequantize_row_iq4_xs_cuda_fp16);
            break;
        case GGML_TYPE_IQ3_S:
            dequantize_cuda_impl<block_iq3_s, QK_K, ggml_half>(
                data, out, dequantize_row_iq3_s_cuda_fp16);
            break;

        default:
            TORCH_CHECK(false, "unsupported quant type");
    }
}
