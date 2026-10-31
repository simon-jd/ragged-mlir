// Comparison-only code: never linked into the MLIR compiler or runtime.
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/version.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <cstddef>
#include <climits>
#include <limits>

static_assert(CUB_VERSION == 300500, "baseline requires CCCL 3.5.0 (environment.lock)");
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

constexpr int kThreads = 256;

// Tree over the row lanes of one feature lane; every thread of the block
// calls it, and every thread gets its own feature lane's result.
__device__ float block_reduce(float value, bool maximum, float *scratch, int lanes) {
  const int row = threadIdx.x / lanes, feature = threadIdx.x % lanes;
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / lanes / 2; stride > 0; stride /= 2) {
    if (row < stride) {
      float other = scratch[threadIdx.x + stride * lanes];
      scratch[threadIdx.x] = maximum ? PropagateMax{}(scratch[threadIdx.x], other)
                                     : scratch[threadIdx.x] + other;
    }
    __syncthreads();
  }
  float result = scratch[feature];
  __syncthreads();
  return result;
}

// One 256-thread block per segment. The threads form `lanes` feature lanes
// (a power of two up to 32, so a warp reads contiguous columns) times
// 256/lanes row lanes: at D=1 all 256 threads stride over the rows. Each
// thread folds its rows (compensated for sums), then a shared-memory tree
// combines the row lanes. Softmax makes three passes: max, sum, normalize.
template <typename Offset>
__global__ void __launch_bounds__(kThreads) block_per_segment(
    const float *values, const Offset *offsets, float *out, int64_t segments, int64_t d, int op, int lanes) {
  __shared__ float scratch[kThreads];
  const int row_lane = threadIdx.x / lanes, row_lanes = kThreads / lanes;
  for (int64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
    const int64_t begin = offsets[segment], end = offsets[segment + 1];
    for (int64_t base = 0; base < d; base += lanes) {
      const int64_t feature = base + threadIdx.x % lanes;
      const bool valid = feature < d;
      float result = op == 0 ? 0.0f : -CUDART_INF_F, carry = 0.0f;
      if (valid) {
        for (int64_t row = begin + row_lane; row < end; row += row_lanes) {
          float x = values[row * d + feature];
          if (op == 0) compensated_add(x, result, carry);
          else result = PropagateMax{}(result, x);
        }
      }
      result = block_reduce(result, op != 0, scratch, lanes);
      if (op != 2) {
        if (valid && row_lane == 0) out[segment * d + feature] = result;
        continue;
      }
      const float maximum = result;
      float total = 0.0f;
      carry = 0.0f;
      if (valid) {
        for (int64_t row = begin + row_lane; row < end; row += row_lanes)
          compensated_add(expf(values[row * d + feature] - maximum), total, carry);
      }
      total = block_reduce(total, false, scratch, lanes);
      if (valid) {
        for (int64_t row = begin + row_lane; row < end; row += row_lanes)
          out[row * d + feature] = expf(values[row * d + feature] - maximum) / total;
      }
    }
  }
}

// The segment holding element i: the largest s with offsets[s] <= i, which
// skips empty segments because offsets[s] == offsets[s + 1] cannot hold i.
template <typename Offset>
__device__ int32_t segment_of(const Offset *offsets, int64_t b, int64_t i) {
  int64_t low = 0, high = b;  // offsets[low] <= i < offsets[high]
  while (high - low > 1) {
    int64_t middle = low + (high - low) / 2;
    if (static_cast<int64_t>(offsets[middle]) <= i) low = middle;
    else high = middle;
  }
  return static_cast<int32_t>(low);
}

// Element-parallel exp(x - max[segment]); also records each element's segment.
template <typename Offset>
__global__ void shifted_exp(const float *input, const Offset *offsets, const float *maximum,
                            float *exp_values, int32_t *segment_ids, int64_t n, int64_t b) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int32_t segment = segment_of(offsets, b, i);
    segment_ids[i] = segment;
    exp_values[i] = expf(input[i] - maximum[segment]);
  }
}

__global__ void normalize(const float *exp_values, const int32_t *segment_ids, const float *total,
                          float *out, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    out[i] = exp_values[i] / total[segment_ids[i]];
}

static unsigned blocks(int64_t b) { return static_cast<unsigned>(b < 65535 ? b : 65535); }
static unsigned element_blocks(int64_t n) {
  int64_t needed = (n + kThreads - 1) / kThreads;
  return static_cast<unsigned>(needed < 4096 ? needed : 4096);
}
static int feature_lanes(int64_t d) {
  int lanes = 1;
  while (lanes < d && lanes < 32) lanes *= 2;
  return lanes;
}
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
  const int lanes = feature_lanes(d);
  if (bits == 32)
    block_per_segment<<<blocks(b), kThreads, 0, s>>>(input, static_cast<const int32_t *>(offsets), out, b, d, op, lanes);
  else
    block_per_segment<<<blocks(b), kThreads, 0, s>>>(input, static_cast<const int64_t *>(offsets), out, b, d, op, lanes);
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
// Softmax at D=1 around two CUB segmented reductions: max (CUB), then
// shifted_exp (element-parallel), sum (CUB) over its output, and normalize.
extern "C" int ragged_cub_shifted_exp(int bits, int64_t b, int64_t n, const float *input,
                                      const void *offsets, const float *maximum, float *exp_values,
                                      int32_t *segment_ids, void *stream) {
  if (!valid(0, bits, b, 1) || n < 0 || b > INT32_MAX) return cudaErrorInvalidValue;
  if (!n) return cudaSuccess;
  auto s = static_cast<cudaStream_t>(stream);
  if (bits == 32)
    shifted_exp<<<element_blocks(n), kThreads, 0, s>>>(input, static_cast<const int32_t *>(offsets), maximum,
                                                       exp_values, segment_ids, n, b);
  else
    shifted_exp<<<element_blocks(n), kThreads, 0, s>>>(input, static_cast<const int64_t *>(offsets), maximum,
                                                       exp_values, segment_ids, n, b);
  return cudaGetLastError();
}
extern "C" int ragged_cub_normalize(int64_t n, const float *exp_values, const int32_t *segment_ids,
                                    const float *total, float *out, void *stream) {
  if (n < 0) return cudaErrorInvalidValue;
  if (!n) return cudaSuccess;
  normalize<<<element_blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(exp_values, segment_ids, total, out, n);
  return cudaGetLastError();
}
