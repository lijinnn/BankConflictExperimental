/**
 * tOsO Shared-Memory Store Experiment - 模拟 epilogue_fwd.hpp 中 tOsO 的拷贝逻辑
 *
 * 编译: nvcc -std=c++17 -O3 -arch=sm_90 --expt-relaxed-constexpr -I magi_attention/csrc/cutlass/include -I magi_attention/csrc/common main.cu -o tOsO_test
 * 运行: ./tOsO_test > test.log 2>&1
 */

#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "magi_attention/csrc/common/utils.h"
#include "magi_attention/csrc/flexible_flash_attention/epilogue_fwd.hpp"

using namespace cute;

// ===================== 配置区域 =====================
static constexpr int kBlockM = 64;     // 可直接改成需要的 blockM
static constexpr int kHeadDim = 128;   // 可直接改成需要的 headDim
static constexpr bool kSwapAB = false; // 如需调试 SwapAB 分支，改为 true
static constexpr int kBlockN = 64;     // 与 epilogue 中一致
static constexpr int kTileSizeBlockM = kBlockM;

using ElementMma = cutlass::half_t; // 主循环使用 16bit
using Element =  float;              // Epilogue 写回使用 32bit
using ElementPartial = float;
using ArchTag = cutlass::arch::Sm90;

// TileShape 定义，与 epilogue 中保持一致
using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
using TileShape_MNK_PV_SwapAB = Shape<Int<kHeadDim>, Int<kBlockM>, Int<kBlockN>>;
using TileShape_MNK_PV_SwapAB_OP_SELECT = Shape<Int<kHeadDim>, Int<kTileSizeBlockM>, Int<kBlockN>>;
using TileShape_MNK_PV_Active = std::conditional_t<kSwapAB, TileShape_MNK_PV_SwapAB_OP_SELECT, TileShape_MNK_PV>;

// Atom layout 选择
using AtomLayoutPV = std::conditional_t<(kBlockM >= 64), Layout<Shape<Int<kBlockM / 64>, _1, _1>>, Layout<Shape<_1, _1, _1>>>;
using AtomLayoutPV_SwapAB = Layout<Shape<_1, Int<kBlockM / kTileSizeBlockM>, _1>>;
using PermutationPV_SwapAB = Tile<Int<kHeadDim>, Int<kBlockM>, Int<kBlockN>>;
// Use if constexpr to avoid instantiating unused PV branches that can trigger static asserts
static constexpr auto make_tiled_mma_pv_active() {
  if constexpr (kSwapAB) {
    return cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<ElementMma, ElementMma, ElementPartial, TileShape_MNK_PV_SwapAB_OP_SELECT, GMMA::Major::MN, GMMA::Major::MN>(),
        AtomLayoutPV_SwapAB{},
        PermutationPV_SwapAB{});
  } else {
    return cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<ElementMma, ElementMma, ElementPartial, TileShape_MNK_PV, GMMA::Major::K, GMMA::Major::MN>(),
        AtomLayoutPV{});
  }
}
using TiledMmaPV = decltype(make_tiled_mma_pv_active());

using ClusterShape = cute::Shape<_1, _1, _1>;
using BlockCoordType = cute::tuple<int, int, int>;
static constexpr int kNumThreads = CUTE_STATIC_V(size(TiledMmaPV{}));
static_assert(kNumThreads <= 1024, "Thread block exceeds CUDA limit");


static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
// static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
// static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
// static constexpr int kSwizzleShift = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
static constexpr int kSwizzle = sizeof(Element) == 4 ? 2 : (kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1)));
static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 3 : (sizeof(Element) == 2 ? 3 : 4);
static constexpr int kSwizzleShift = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
using SmemLayoutAtomOTMA = decltype(cutlass::gemm::collective::detail::
                                        ss_smem_selector<cute::GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK_PV{})), decltype(cute::get<1>(TileShape_MNK_PV{}))>());
using SmemLayoutOTMA = decltype(tile_to_shape(SmemLayoutAtomOTMA{}, select<0, 1>(TileShape_MNK_PV{})));
using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleShift>{}, Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
// static constexpr int swizzle_B = 2;
// static constexpr int swizzle_M = 2;
// static constexpr int swizzle_S = 2;
// using SmemLayoutAtomO = decltype(composition(Swizzle<swizzle_B, swizzle_M, swizzle_S>{}, Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
using SmemLayoutOSTS = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_MNK_PV{})));
// using SmemLayoutO = std::conditional_t<ArchTag::kMinComputeCapability >= 90, SmemLayoutOTMA, SmemLayoutOSTS>;
using SmemLayoutO = SmemLayoutOSTS;

static constexpr int kSmemElements = CUTE_STATIC_V(size(SmemLayoutO{}));
static constexpr size_t kSharedMemBytes = sizeof(Element) * kSmemElements * 2;

template <class Layout>
CUTE_HOST_DEVICE auto get_layout_stride(Layout const& layout) {
  if constexpr (cute::is_composed_layout<Layout>::value) {
    return layout.layout_b().stride();
  } else {
    return layout.stride();
  }
}

__global__ void visualize_tOsO_kernel() {
  if (threadIdx.x == 0) {
    printf("kThreadNum: %d\n", kNumThreads);
  }
  extern __shared__ char smem_raw[];
  Element* sO_thread_ptr = reinterpret_cast<Element*>(smem_raw);
  Element* sO_reg_ptr = sO_thread_ptr + kSmemElements;

  auto sO_thread = make_tensor(make_smem_ptr(sO_thread_ptr), SmemLayoutO{});
  auto sO_reg = make_tensor(make_smem_ptr(sO_reg_ptr), SmemLayoutO{});

  TiledMmaPV tiled_mma;
  auto tiled_copy_O = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>{}, tiled_mma);
  if (threadIdx.x == 0) {
    print("========== SmemLayoutO ==========\n");
    print(SmemLayoutO{});
    print("\n");
    printf("\n=== Tiled Copy 信息 ===\n");
    print("tiled_copy_O: ");
    print(tiled_copy_O);
    print("\n");
  }

  int thread_idx = threadIdx.x;
  auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
  auto thr_copy_O = tiled_copy_O.get_thread_slice(thread_idx);

  Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK_PV_Active{}));
  Tensor tOrFinalO = make_tensor_like<Element>(tOrO);
  Tensor fragment_coords = make_identity_tensor(shape(tOrO));
  Tensor cO = make_identity_tensor(select<0, 1>(TileShape_MNK_PV_Active{}));
  Tensor tOcO = thr_mma.partition_C(cO);

  auto make_dest_tensor = [&](auto& sO_tensor) {
    if constexpr (!kSwapAB) {
      if (threadIdx.x == 0) {
        printf("sO_tensor layout: ");
        print(sO_tensor.layout());
        printf("\n");
      }
      return thr_copy_O.partition_D(sO_tensor);
    } else {
      auto sO_layout = sO_tensor.layout();
      auto sO_shape = sO_layout.shape();
      auto sO_stride = get_layout_stride(sO_layout);
      auto sO_transposed = make_tensor(
          sO_tensor.data(),
          cute::make_layout(
              cute::make_shape(get<1>(sO_shape), get<0>(sO_shape)),
              cute::make_stride(get<1>(sO_stride), get<0>(sO_stride))));
      return thr_copy_O.partition_D(sO_transposed);
    }
  };

  auto store_identity = [&](auto& sO_tensor, bool use_thread_id) {
    // if (use_thread_id && thread_idx < 32) {
    //   auto layout = tOrO.layout();
    //   for (int i = 0; i < size(tOrO); ++i) {
    //     auto frag_coord = fragment_coords(i);
    //     int reg_idx = int(layout(frag_coord));
    //     auto mn_coord = tOcO(i);
    //     int row = int(get<0>(mn_coord));
    //     int col = int(get<1>(mn_coord));
    //     printf("  T%-3d Reg%-3d -> Row %-3d Col %-3d\n", thread_idx, reg_idx, row, col);
    //   }
    // }
    for (int i = 0; i < size(tOrO); ++i) {
      ElementPartial value = use_thread_id ? ElementPartial(thread_idx) : ElementPartial(i);
      tOrO(i) = value;
    }
    cutlass::NumericConverter<Element, ElementPartial> convert_op;
#pragma unroll
    for (int i = 0; i < size(tOrO); ++i) {
      tOrFinalO(i) = convert_op(tOrO(i));
    }
    auto tOrO_copy_view = thr_copy_O.retile_S(tOrFinalO);
    auto tOsO = make_dest_tensor(sO_tensor);
    constexpr int kPrintThreadIdx =  0;
    if (use_thread_id && thread_idx == kPrintThreadIdx) {
      printf("\n=== tOsO Layout 信息 ===\n");
      print("tOrO layout: ");
      print(tOrO.layout());
      print("\n");
      print("tOrFinalO layout: ");
      print(tOrFinalO.layout());
      print("\n");
      print("tOrO_copy_view layout: ");
      print(tOrO_copy_view.layout());
      print("\n");
      print("tOsO layout: ");
      print(tOsO.layout());
      print("\n");
    }
    cute::copy(tiled_copy_O, tOrO_copy_view, tOsO);
  };

  store_identity(sO_thread, true);
  __syncthreads();
  store_identity(sO_reg, false);
  __syncthreads();

  if (thread_idx == 0) {
    auto decode_half_to_int = [](Element value) {
      return static_cast<int>(static_cast<float>(value));
    };

    printf("==============================\n");
    printf("tOsO store visualization\n");
    printf("SwapAB: %s, kBlockM: %d, headDim: %d (BlockN fixed to %d)\n", kSwapAB ? "true" : "false", kBlockM, kHeadDim, kBlockN);
    printf("Smem elements per tensor: %d (shared bytes per tensor %zu)\n", kSmemElements, sizeof(Element) * kSmemElements);

    constexpr int kMaxRowsPrint = 16;
    constexpr int kMaxColsPrint = 8;
    int rows_to_print_matrix = kBlockM < kMaxRowsPrint ? kBlockM : kMaxRowsPrint;
    int cols_to_print_matrix = kHeadDim < kMaxColsPrint ? kHeadDim : kMaxColsPrint;

    printf("\n矩阵视图（前 %d 行，前 %d 列）\n", rows_to_print_matrix, cols_to_print_matrix);
    printf("Row/Col -> T(thread) | R(reg) | I(index) | B(bank)\n");
    for (int row = 0; row < rows_to_print_matrix; ++row) {
      printf("Row %2d: ", row);
      for (int col = 0; col < cols_to_print_matrix; ++col) {
        int thread_val = decode_half_to_int(sO_thread(row, col));
        int reg_val = decode_half_to_int(sO_reg(row, col));
        int linear_idx = static_cast<int>(sO_thread.layout()(make_coord(row, col)));
        int offset_bytes = linear_idx * int(sizeof(Element));
        int bank = (offset_bytes / 4) % 32;
        printf("[T%3d R%3d I%4d B%2d] ", thread_val, reg_val, linear_idx, bank);
      }
      printf("\n");
    }

    printf("\nLinear shared-memory view (first 128 slots)\n");
    int linear_limit = kSmemElements < 128 ? kSmemElements : 128;
    for (int idx = 0; idx < linear_limit; ++idx) {
      int thread_val = decode_half_to_int(sO_thread_ptr[idx]);
      int reg_val = decode_half_to_int(sO_reg_ptr[idx]);
      int offset_bytes = idx * int(sizeof(Element));
      int bank = (offset_bytes / 4) % 32;
      printf("[%03d] offset=%4d bytes bank=%02d  thread=%4d  reg=%4d\n", idx, offset_bytes, bank, thread_val, reg_val);
    }

    printf("\n=== Bank Conflict 分析（按 warp / 寄存器）===\n");
    struct AddrInfo {
      int thread_id;
      int reg_idx;
      int bank_id;
      bool valid;
    };

    int total_elems = kSmemElements;
    AddrInfo* addr_map = new AddrInfo[total_elems];

    for (int idx = 0; idx < total_elems; ++idx) {
      int thread_val = decode_half_to_int(sO_thread_ptr[idx]);
      int reg_val = decode_half_to_int(sO_reg_ptr[idx]);
      int offset_bytes = idx * int(sizeof(Element));
      int bank = (offset_bytes / 4) % 32;
      addr_map[idx].thread_id = thread_val;
      addr_map[idx].reg_idx = reg_val;
      addr_map[idx].bank_id = bank;
      addr_map[idx].valid = (thread_val >= 0 && thread_val < kNumThreads && reg_val >= 0);
    }

    int max_reg_idx = -1;
    for (int idx = 0; idx < total_elems; ++idx) {
      if (addr_map[idx].valid && addr_map[idx].reg_idx > max_reg_idx) {
        max_reg_idx = addr_map[idx].reg_idx;
      }
    }
    int num_regs_per_thread = max_reg_idx + 1;
    int warps_total = (kNumThreads + 31) / 32;

    printf("检测到每个线程最多 %d 个寄存器写入槽位\n", num_regs_per_thread);
    printf("共 %d 个 warp 参与拷贝\n\n", warps_total);

    for (int warp_id = 0; warp_id < warps_total; ++warp_id) {
      int thread_start = warp_id * 32;
      int thread_end = thread_start + 31;
      if (thread_start >= kNumThreads) {
        continue;
      }
      if (thread_end >= kNumThreads) {
        thread_end = kNumThreads - 1;
      }

      printf("=== Warp %d (Threads %d-%d) ===\n", warp_id, thread_start, thread_end);

      for (int reg_idx = 0; reg_idx < num_regs_per_thread; ++reg_idx) {
        printf("\n寄存器槽位 %d:\n", reg_idx);

        int bank_histogram[32] = {0};
        int lane_to_bank[32];
        int lane_to_addr[32];
        for (int lane = 0; lane < 32; ++lane) {
          lane_to_bank[lane] = -1;
          lane_to_addr[lane] = -1;
        }

        for (int idx = 0; idx < total_elems; ++idx) {
          if (!addr_map[idx].valid) {
            continue;
          }
          int tid = addr_map[idx].thread_id;
          int rid = addr_map[idx].reg_idx;
          if (tid < thread_start || tid > thread_end || rid != reg_idx) {
            continue;
          }
          int lane = tid - thread_start;
          if (lane < 0 || lane >= 32) {
            continue;
          }
          int bank = addr_map[idx].bank_id;
          if (lane_to_bank[lane] == -1) {
            lane_to_bank[lane] = bank;
            lane_to_addr[lane] = idx;
            bank_histogram[bank]++;
          } else {
            printf("  冲突: T%-3d -> addr[%4d] -> bank %2d (已占用 addr[%4d])\n", tid, idx, bank, lane_to_addr[lane]);
            bank_histogram[bank]++;
          }
        }

        int active_threads = 0;
        for (int lane = 0; lane < 32; ++lane) {
          if (lane_to_addr[lane] >= 0) {
            active_threads++;
          }
        }

        if (active_threads == 0) {
          printf("  (该 warp 在此寄存器槽位无写入)\n");
          continue;
        }

        printf("  线程 -> 地址 -> Bank:\n    ");
        for (int lane = 0; lane < 32 && lane < (thread_end - thread_start + 1); ++lane) {
          if (lane_to_addr[lane] >= 0) {
            int tid = thread_start + lane;
            printf("T%-3d->[%4d]->B%-2d  ", tid, lane_to_addr[lane], lane_to_bank[lane]);
          }
          if ((lane + 1) % 4 == 0) {
            printf("\n    ");
          }
        }
        printf("\n");

        int max_conflicts = 0;
        int conflict_banks = 0;
        for (int b = 0; b < 32; ++b) {
          if (bank_histogram[b] > max_conflicts) {
            max_conflicts = bank_histogram[b];
          }
          if (bank_histogram[b] > 1) {
            conflict_banks++;
          }
        }

        if (max_conflicts > 1) {
          printf("  ⚠️  检测到 %d 路 bank 冲突，发生在 %d 个 bank 上\n", max_conflicts, conflict_banks);
        } else {
          printf("  ✓  无 bank 冲突\n");
        }
      }

      printf("\n");
    }

    delete[] addr_map;
    printf("==============================\n\n");
  }
}

int main() {
  std::cout << "=== tOsO 拷贝实验 ===" << std::endl;
  std::cout << "模拟 epilogue_fwd.hpp 中的 shared memory 写回逻辑\n" << std::endl;

  std::cout << "配置参数:" << std::endl;
  std::cout << "  SwapAB        = " << (kSwapAB ? "true" : "false") << std::endl;
  std::cout << "  kBlockM       = " << kBlockM << std::endl;
  std::cout << "  kHeadDim      = " << kHeadDim << std::endl;
  std::cout << "  kBlockN       = " << kBlockN << std::endl;
  std::cout << "  TileSizeBlockM= " << kTileSizeBlockM << std::endl;
  std::cout << "  NumThreads    = " << kNumThreads << std::endl;
  std::cout << "  SharedMem/tensor = " << sizeof(Element) * kSmemElements << " bytes" << std::endl;
  std::cout << "  SharedMem total  = " << kSharedMemBytes << " bytes" << std::endl;

  cudaError_t attr_err = cudaFuncSetAttribute(visualize_tOsO_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedMemBytes);
  if (attr_err != cudaSuccess) {
    std::cerr << "设置 kernel 动态 shared memory 失败: " << cudaGetErrorString(attr_err) << std::endl;
    return 1;
  }

  visualize_tOsO_kernel<<<1, kNumThreads, kSharedMemBytes>>>();
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::cerr << "kernel launch 失败: " << cudaGetErrorString(launch_err) << std::endl;
    return 1;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA 错误: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  return 0;
}

