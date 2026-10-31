// Comparison-only code: never linked into the MLIR compiler or runtime.
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/version.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <cstddef>
#include <climits>
#include <limits>

static_assert(CUB_VERSION == 300500, "baseline requires the plan-pinned CCCL 3.5.0");
struct Add {
  __host__ __device__ float operator()(float a, float b) const { return a + b; }
};
struct PropagateMax {
  __host__ __device__ float operator()(float a, float b) const {
    return a != a ? a : b != b ? b : a > b ? a : b;
  }
};
__device__ void compensated_add(float x, float &sum, float &carry) {
  float y = x - carry;
  float next = sum + y;
  carry = isfinite(next) ? (next - sum) - y : 0.0f;
  sum = next;
}

template <typename Offset>
__global__ void naive(const float *values, const Offset *offsets, float *out,
                      int64_t segments, int64_t d, int op) {
  for (int64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
    const int64_t begin = offsets[segment], end = offsets[segment + 1];
    for (int64_t feature = threadIdx.x; feature < d; feature += blockDim.x) {
      float result = op == 0 ? 0.0f : -CUDART_INF_F;
      float carry = 0.0f;
      for (int64_t row = begin; row < end; ++row) {
        float x = values[row * d + feature];
        if (op == 0) compensated_add(x, result, carry);
        else result = PropagateMax{}(result, x);
      }
      if (op != 2) {
        out[segment * d + feature] = result;
        continue;
      }
      float total = 0.0f;
      carry = 0.0f;
      for (int64_t row = begin; row < end; ++row)
        compensated_add(expf(values[row * d + feature] - result), total, carry);
      for (int64_t row = begin; row < end; ++row)
        out[row * d + feature] = expf(values[row * d + feature] - result) / total;
    }
  }
}

template <typename Offset, bool Normalize>
__global__ void transform(const float *input, const Offset *offsets,
                          const float *stats, float *out, int64_t segments) {
  for (int64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
    const int64_t begin = offsets[segment], end = offsets[segment + 1];
    for (int64_t row = begin + threadIdx.x; row < end; row += blockDim.x)
      out[row] = Normalize ? input[row] / stats[segment]
                           : expf(input[row] - stats[segment]);
  }
}

static unsigned blocks(int64_t b) { return static_cast<unsigned>(b < 65535 ? b : 65535); }
static bool valid(int op, int bits, int64_t b, int64_t d) {
  return op >= 0 && op <= 2 && (bits == 32 || bits == 64) && b >= 0 && d >= 1 && d <= 1024;
}
extern "C" int ragged_baseline_version() { return CUB_VERSION; }
extern "C" const char *ragged_baseline_error(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}
extern "C" int ragged_naive(int op, int bits, int64_t b, int64_t d,
                            const float *input, const void *offsets, float *out, void *stream) {
  if (!valid(op, bits, b, d)) return cudaErrorInvalidValue;
  if (!b) return cudaSuccess;
  auto s = static_cast<cudaStream_t>(stream);
  if (bits == 32) naive<<<blocks(b), 128, 0, s>>>(input, static_cast<const int32_t *>(offsets), out, b, d, op);
  else naive<<<blocks(b), 128, 0, s>>>(input, static_cast<const int64_t *>(offsets), out, b, d, op);
  return cudaGetLastError();
}

template <typename Offset>
cudaError_t reduce(int op, void *scratch, size_t &bytes, const float *input,
                   const Offset *offsets, float *out, int64_t b, cudaStream_t stream) {
  if (op == 0)
    return cub::DeviceSegmentedReduce::Reduce(scratch, bytes, input, out, b,
                                              offsets, offsets + 1, Add{}, 0.0f, stream);
  return cub::DeviceSegmentedReduce::Reduce(scratch, bytes, input, out, b,
                                            offsets, offsets + 1, PropagateMax{}, -std::numeric_limits<float>::infinity(), stream);
}
extern "C" int ragged_cub_reduce(int op, int bits, int64_t b, const float *input,
                                 const void *offsets, float *out, void *scratch,
                                 size_t *bytes, void *stream) {
  if (!valid(op, bits, b, 1) || op == 2 || !bytes) return cudaErrorInvalidValue;
  if (!b) { *bytes = 0; return cudaSuccess; }
  auto s = static_cast<cudaStream_t>(stream);
  if (bits == 32) return reduce(op, scratch, *bytes, input, static_cast<const int32_t *>(offsets), out, b, s);
  return reduce(op, scratch, *bytes, input, static_cast<const int64_t *>(offsets), out, b, s);
}
extern "C" int ragged_cub_transform(int normalize, int bits, int64_t b,
                                    const float *input, const void *offsets,
                                    const float *stats, float *out, void *stream) {
  if ((normalize != 0 && normalize != 1) || !valid(0, bits, b, 1)) return cudaErrorInvalidValue;
  if (!b) return cudaSuccess;
  auto s = static_cast<cudaStream_t>(stream);
  if (bits == 32) {
    if (normalize) transform<int32_t, true><<<blocks(b), 128, 0, s>>>(input, static_cast<const int32_t *>(offsets), stats, out, b);
    else transform<int32_t, false><<<blocks(b), 128, 0, s>>>(input, static_cast<const int32_t *>(offsets), stats, out, b);
  } else {
    if (normalize) transform<int64_t, true><<<blocks(b), 128, 0, s>>>(input, static_cast<const int64_t *>(offsets), stats, out, b);
    else transform<int64_t, false><<<blocks(b), 128, 0, s>>>(input, static_cast<const int64_t *>(offsets), stats, out, b);
  }
  return cudaGetLastError();
}
