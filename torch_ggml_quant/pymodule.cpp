#include "quant.h"
#include "dequant.h"

#include "quant_formats.h"


int quant_format_block_size(int quant_type) {
    switch (quant_type) {
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
        case GGML_TYPE_Q4_1: return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);

        case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_Q8_K: return sizeof(block_q8_K);

        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS: return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ1_S: return sizeof(block_iq1_s);
        case GGML_TYPE_IQ1_M: return sizeof(block_iq1_m);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
        case GGML_TYPE_IQ4_XS: return sizeof(block_iq4_xs);
        case GGML_TYPE_IQ3_S: return sizeof(block_iq3_s);

        default:
            TORCH_CHECK(false, "unsupported quant type");
    }
}

int quant_format_values_per_block(int quant_type) {
    switch (quant_type) {
        case GGML_TYPE_Q4_0: return QK4_0;
        case GGML_TYPE_Q4_1: return QK4_1;
        case GGML_TYPE_Q5_0: return QK5_0;
        case GGML_TYPE_Q5_1: return QK5_1;
        case GGML_TYPE_Q8_0: return QK8_0;

        case GGML_TYPE_Q2_K: return QK_K;
        case GGML_TYPE_Q3_K: return QK_K;
        case GGML_TYPE_Q4_K: return QK_K;
        case GGML_TYPE_Q5_K: return QK_K;
        case GGML_TYPE_Q6_K: return QK_K;
        case GGML_TYPE_Q8_K: return QK_K;

        case GGML_TYPE_IQ2_XXS: return QK_K;
        case GGML_TYPE_IQ2_XS: return QK_K;
        case GGML_TYPE_IQ2_S: return QK_K;
        case GGML_TYPE_IQ3_XXS: return QK_K;
        case GGML_TYPE_IQ1_S: return QK_K;
        case GGML_TYPE_IQ1_M: return QK_K;
        case GGML_TYPE_IQ4_NL: return QK4_NL;
        case GGML_TYPE_IQ4_XS: return QK_K;
        case GGML_TYPE_IQ3_S: return QK_K;

        default:
            TORCH_CHECK(false, "unsupported quant type");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_fp32_cpu", &quantize_fp32_cpu, "quantize_fp32_cpu");
    m.def("dequantize_fp32_cpu", &dequantize_fp32_cpu, "dequantize_fp32_cpu");
    m.def("dequantize_fp32_cuda", &dequantize_fp32_cuda, "dequantize_fp32_cuda");
    m.def("dequantize_fp16_cuda", &dequantize_fp16_cuda, "dequantize_fp16_cuda");
    m.def("quant_format_block_size", &quant_format_block_size, "quant_format_block_size");
    m.def("quant_format_values_per_block", &quant_format_values_per_block,
        "quant_format_values_per_block");
}
