// MXFP4 (E2M1 + E8M0) Fused GPU Kernels for KV Cache Compression
//
// Dequant: nibble unpack → E2M1 LUT → E8M0 scale mul → bf16
// Quant:   bf16 → block max → E8M0 scale → E2M1 searchsorted → nibble pack
//
// Format: OCP MX with block_size=16
//   Data:  4-bit E2M1 values packed as nibble pairs (2 per uint8 byte)
//   Scale: 8-bit E8M0 exponent per 16-element block (value = 2^(byte - 127))
//
// E2M1 encoding (3-bit magnitude):
//   e=0 (subnorm): 0.0, 0.5
//   e=1:           1.0, 1.5
//   e=2:           2.0, 3.0
//   e=3:           4.0, 6.0
//   4th bit: sign

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

constexpr int MX_BLOCK_SIZE = 16;
constexpr int THREADS = 256;


// ─── Dequantize Kernel ───
// One block per token, each thread processes packed bytes at stride THREADS.
// For N=576 (MLA): 288 packed bytes, ~2 iterations per thread.
__global__ void mxfp4_dequant_kernel(
    const uint8_t* __restrict__ packed,     // [T, N_half]
    const uint8_t* __restrict__ scales,     // [T, N_blocks]
    __hip_bfloat16* __restrict__ output,    // [T, N]
    int T, int N_half)                      // N_half = N/2
{
    const int token = blockIdx.x;
    if (token >= T) return;
    const int tid = threadIdx.x;
    const int N_blocks = N_half / 8;  // N/16

    const uint8_t* my_packed = packed + (int64_t)token * N_half;
    const uint8_t* my_scales = scales + (int64_t)token * N_blocks;
    __hip_bfloat16* my_out = output + (int64_t)token * (N_half * 2);

    // E2M1 full LUT in LDS: nibble → float (sign + magnitude fused)
    __shared__ float s_lut[16];
    if (tid < 16) {
        constexpr float lut[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,     // +
            0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f // −
        };
        s_lut[tid] = lut[tid];
    }
    __syncthreads();

    for (int byte_idx = tid; byte_idx < N_half; byte_idx += blockDim.x) {
        uint8_t packed_byte = my_packed[byte_idx];

        float lo_val = s_lut[packed_byte & 0x0F];
        float hi_val = s_lut[(packed_byte >> 4) & 0x0F];

        // E8M0 block scale: 2^(scale_byte - 127)
        // 8 packed bytes (16 elements) share one scale
        float scale = exp2f((float)my_scales[byte_idx / 8] - 127.0f);

        int out_idx = byte_idx * 2;
        my_out[out_idx]     = __float2bfloat16(lo_val * scale);
        my_out[out_idx + 1] = __float2bfloat16(hi_val * scale);
    }
}


// ─── Quantize Kernel ───
// Two-phase: (1) block maxima → E8M0 scales, (2) quantize + nibble pack.
// Shared memory holds per-block scale exponents between phases.
__global__ void mxfp4_quant_kernel(
    const __hip_bfloat16* __restrict__ input,  // [T, N]
    uint8_t* __restrict__ packed,               // [T, N_half]
    uint8_t* __restrict__ scales,               // [T, N_blocks]
    int T, int N, int N_half, int N_blocks)
{
    const int token = blockIdx.x;
    if (token >= T) return;
    const int tid = threadIdx.x;

    const __hip_bfloat16* my_in = input + (int64_t)token * N;
    uint8_t* my_packed = packed + (int64_t)token * N_half;
    uint8_t* my_scales = scales + (int64_t)token * N_blocks;

    extern __shared__ float s_scale_exp[];

    // Phase 1: block maxima → E8M0 scale exponents
    for (int blk = tid; blk < N_blocks; blk += blockDim.x) {
        float max_abs = 0.0f;
        int base = blk * MX_BLOCK_SIZE;
        #pragma unroll
        for (int j = 0; j < MX_BLOCK_SIZE; j++) {
            int elem = base + j;
            if (elem < N) {
                float v = fabsf(__bfloat162float(my_in[elem]));
                max_abs = fmaxf(max_abs, v);
            }
        }
        float se = ceilf(log2f(fmaxf(max_abs / 6.0f, 1e-10f)));
        my_scales[blk] = (uint8_t)fminf(fmaxf(se + 127.0f, 0.0f), 255.0f);
        s_scale_exp[blk] = se;
    }
    __syncthreads();

    // Phase 2: quantize + nibble pack
    for (int byte_idx = tid; byte_idx < N_half; byte_idx += blockDim.x) {
        int elem0 = byte_idx * 2;
        int elem1 = elem0 + 1;
        int blk = elem0 / MX_BLOCK_SIZE;
        float inv_scale = exp2f(-s_scale_exp[blk]);

        // Element 0
        float v0 = __bfloat162float(my_in[elem0]) * inv_scale;
        uint8_t sign0 = (v0 < 0.0f) ? 0x08u : 0x00u;
        float a0 = fabsf(v0);
        uint8_t mag0 = (uint8_t)(
            (a0 >= 0.25f) + (a0 >= 0.75f) + (a0 >= 1.25f) +
            (a0 >= 1.75f) + (a0 >= 2.5f) + (a0 >= 3.5f) + (a0 >= 5.0f));

        // Element 1
        uint8_t nibble1 = 0;
        if (elem1 < N) {
            float v1 = __bfloat162float(my_in[elem1]) * inv_scale;
            uint8_t sign1 = (v1 < 0.0f) ? 0x08u : 0x00u;
            float a1 = fabsf(v1);
            uint8_t mag1 = (uint8_t)(
                (a1 >= 0.25f) + (a1 >= 0.75f) + (a1 >= 1.25f) +
                (a1 >= 1.75f) + (a1 >= 2.5f) + (a1 >= 3.5f) + (a1 >= 5.0f));
            nibble1 = sign1 | mag1;
        }

        my_packed[byte_idx] = (sign0 | mag0) | (nibble1 << 4);
    }
}


// ─── Python Bindings ───

torch::Tensor mxfp4_dequantize(
    torch::Tensor packed, torch::Tensor scales, int64_t N)
{
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kUInt8, "scales must be uint8");
    TORCH_CHECK(N % MX_BLOCK_SIZE == 0, "N must be a multiple of ", MX_BLOCK_SIZE);

    auto packed_contig = packed.contiguous();
    auto scales_contig = scales.contiguous();

    auto orig_sizes = packed.sizes().vec();
    auto packed_flat = packed_contig.reshape({-1, N / 2});
    auto scales_flat = scales_contig.reshape({-1, N / MX_BLOCK_SIZE});

    const int T = packed_flat.size(0);
    const int N_half = N / 2;
    TORCH_CHECK(packed_flat.size(1) == N_half,
        "packed last dim should be ", N_half, ", got ", packed_flat.size(1));
    TORCH_CHECK(scales_flat.size(1) == N / MX_BLOCK_SIZE,
        "scales last dim should be ", N / MX_BLOCK_SIZE, ", got ", scales_flat.size(1));

    auto output = torch::empty({T, N},
        torch::dtype(torch::kBFloat16).device(packed.device()));
    if (T == 0) {
        auto out_sizes = orig_sizes;
        out_sizes.back() = N;
        return output.reshape(out_sizes);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    mxfp4_dequant_kernel<<<T, THREADS, 0, stream>>>(
        packed_flat.data_ptr<uint8_t>(),
        scales_flat.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
        T, N_half);

    auto out_sizes = orig_sizes;
    out_sizes.back() = N;
    return output.reshape(out_sizes);
}

void mxfp4_dequantize_inplace(
    torch::Tensor packed, torch::Tensor scales, torch::Tensor output)
{
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(scales.scalar_type() == torch::kUInt8);
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16);

    const int64_t N = output.size(-1);
    TORCH_CHECK(N % MX_BLOCK_SIZE == 0);

    auto packed_flat = packed.contiguous().reshape({-1, N / 2});
    auto scales_flat = scales.contiguous().reshape({-1, N / MX_BLOCK_SIZE});
    auto output_flat = output.reshape({-1, N});

    const int T = packed_flat.size(0);
    if (T == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream();
    mxfp4_dequant_kernel<<<T, THREADS, 0, stream>>>(
        packed_flat.data_ptr<uint8_t>(),
        scales_flat.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(output_flat.data_ptr()),
        T, (int)(N / 2));
}

std::vector<torch::Tensor> mxfp4_quantize(torch::Tensor tensor)
{
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, "input must be bf16");
    const int64_t N = tensor.size(-1);
    TORCH_CHECK(N % MX_BLOCK_SIZE == 0, "N must be a multiple of ", MX_BLOCK_SIZE);

    auto orig_sizes = tensor.sizes().vec();
    auto input_flat = tensor.contiguous().reshape({-1, N});
    const int T = input_flat.size(0);
    const int N_half = N / 2;
    const int N_blocks = N / MX_BLOCK_SIZE;

    auto packed = torch::empty({T, N_half},
        torch::dtype(torch::kUInt8).device(tensor.device()));
    auto scales = torch::empty({T, N_blocks},
        torch::dtype(torch::kUInt8).device(tensor.device()));

    if (T > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        int smem = N_blocks * sizeof(float);
        mxfp4_quant_kernel<<<T, THREADS, smem, stream>>>(
            reinterpret_cast<const __hip_bfloat16*>(input_flat.data_ptr()),
            packed.data_ptr<uint8_t>(),
            scales.data_ptr<uint8_t>(),
            T, (int)N, N_half, N_blocks);
    }

    auto packed_sizes = orig_sizes;
    packed_sizes.back() = N_half;
    auto scale_sizes = orig_sizes;
    scale_sizes.back() = N_blocks;

    return {packed.reshape(packed_sizes), scales.reshape(scale_sizes)};
}

void mxfp4_quantize_inplace(
    torch::Tensor tensor, torch::Tensor packed_out, torch::Tensor scales_out)
{
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16);
    const int64_t N = tensor.size(-1);
    TORCH_CHECK(N % MX_BLOCK_SIZE == 0);

    auto input_flat = tensor.contiguous().reshape({-1, N});
    auto packed_flat = packed_out.reshape({-1, N / 2});
    auto scales_flat = scales_out.reshape({-1, N / MX_BLOCK_SIZE});

    const int T = input_flat.size(0);
    if (T == 0) return;

    const int N_half = N / 2;
    const int N_blocks = N / MX_BLOCK_SIZE;
    auto stream = at::cuda::getCurrentCUDAStream();
    int smem = N_blocks * sizeof(float);

    mxfp4_quant_kernel<<<T, THREADS, smem, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(input_flat.data_ptr()),
        packed_flat.data_ptr<uint8_t>(),
        scales_flat.data_ptr<uint8_t>(),
        T, (int)N, N_half, N_blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_dequantize", &mxfp4_dequantize,
          "MXFP4 E2M1 dequantization (fused nibble unpack + LUT + E8M0 scale)");
    m.def("mxfp4_dequantize_inplace", &mxfp4_dequantize_inplace,
          "MXFP4 E2M1 dequantization in-place (CUDA graph compatible)");
    m.def("mxfp4_quantize", &mxfp4_quantize,
          "MXFP4 E2M1 quantization (fused block max + E8M0 scale + nibble pack)");
    m.def("mxfp4_quantize_inplace", &mxfp4_quantize_inplace,
          "MXFP4 E2M1 quantization in-place (CUDA graph compatible)");
}
