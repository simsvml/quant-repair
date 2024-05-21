#include <torch/extension.h>

#include <iostream>

#include <ggml-common.h>
#include <ggml-quants.h>

torch::Tensor dequant_q6_K(torch::Tensor data) {
    TORCH_CHECK(data.sizes().size() == 1, "data must have exactly 1 dimension");
    TORCH_CHECK(data.device().is_cpu(), "data must be a cpu tensor");
    size_t quant_size = data.sizes()[0];
    TORCH_CHECK(quant_size % sizeof(block_q6_K) == 0,
        "data size must be a multiple of sizeof(block_q6_K)");
    size_t num_blocks = quant_size / sizeof(block_q6_K);
    auto out_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU);
    auto out = torch::ones({(ssize_t)num_blocks * QK_K}, out_options);

    const void* data_ptr = data.const_data_ptr();
    TORCH_CHECK((uintptr_t)data_ptr % alignof(block_q6_K) == 0,
        "data ptr is misaligned for block_q6_K");
    const block_q6_K* data_block_ptr = (const block_q6_K*)data_ptr;

    float* out_ptr = out.mutable_data_ptr<float>();
    std::cout << "out ptr = " << (void*)out_ptr << "\n";

    dequantize_row_q6_K(data_block_ptr, out_ptr, num_blocks * QK_K);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    // Need to call `ggml_init` once to initialize some lookup tables used in
    // dequantize functions.
    struct ggml_init_params params = {
        /*.mem_size   =*/ 0,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_free(ctx);

    m.def("dequant_q6_K", &dequant_q6_K, "dequant_q6_K");
}
