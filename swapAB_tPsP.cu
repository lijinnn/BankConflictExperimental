/**
 * CUTE P矩阵拷贝实验 - 模拟 mainloop_fwd_sm90_tma_gmma_ws.hpp 中的 P 拷贝逻辑
 * 
 * 编译: nvcc -std=c++17 -O3 -arch=sm_90 -I magi_attention/csrc/cutlass/include -I magi_attention/csrc/common --expt-relaxed-constexpr --expt-extended-lambda swapAB_tPsP.cu -o swapAB_tPsP_test
 * 运行: ./swapAB_tPsP_test 2&>1 > test.log
 * 
 * DEBUG 开关说明：
 *   - 定义 DEBUG_PRINT：kernel 会输出详细的调试信息
 *   - 注释掉 DEBUG_PRINT：kernel 不会输出任何信息（只保留 main 函数的输出）
 */

// DEBUG 开关：注释掉下面这行可关闭 kernel 内的所有输出
#define DEBUG_PRINT

#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <cute/tensor.hpp>
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "utils.h"

using namespace cute;

// 配置参数，模拟 FlashAttention 的设置
static constexpr int kBlockM = 8;
static constexpr int kBlockN = 64;
static constexpr int kHeadDim = 128;
static constexpr bool SwapAB = true;  // 开启 SwapAB 模式
static constexpr int TileSize_kBlockM = kBlockM;  // 或者 kBlockM / 2

// 定义 TileShape
using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

// SwapAB 模式下的 TileShape
using TileShape_MNK_SwapAB = Shape<Int<kBlockN>, Int<kBlockM>, Int<kHeadDim>>;
using TileShape_MNK_SwapAB_OP_SELECT = Shape<Int<kBlockN>, Int<TileSize_kBlockM>, Int<kHeadDim>>;

// 定义 Element 类型
using Element = cutlass::half_t;
using IdentityElement = uint16_t;
static_assert(sizeof(IdentityElement) == sizeof(Element), "IdentityElement size mismatch");

// 定义 SmemLayoutP，根据 SwapAB 模式选择不同的 Major
// 当 SwapAB = true 时，使用 GMMA::Major::MN；否则使用 GMMA::Major::K
using SmemLayoutAtomP = std::conditional_t<
    !SwapAB,
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, 
        Element, 
        decltype(cute::get<0>(TileShape_MNK{})), 
        decltype(cute::get<1>(TileShape_MNK{}))
    >()),
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::MN, 
        Element, 
        decltype(cute::get<0>(TileShape_MNK{})), 
        decltype(cute::get<1>(TileShape_MNK{}))
    >())
>;
using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));

// 定义 SmemCopyAtomP
using SmemCopyAtomP = std::conditional_t<
    TileSize_kBlockM == 8,
    Copy_Atom<cute::SM90_U32x2_STSM_N, Element>,
    Copy_Atom<cute::SM90_U32x4_STSM_N, Element>
>;

// 模拟 TiledMma，根据 SwapAB 模式选择不同的配置
// 定义两种 AtomLayout
// 当 kBlockM < 64 时，AtomLayoutQK 使用占位符（因为不会在 SwapAB=true 时使用）
using AtomLayoutQK = Layout<Shape<_1, _1, _1>>;
using AtomLayoutQK_SwapAB = Layout<Shape<_1, Int<kBlockM / TileSize_kBlockM>, _1>>;

// 为了避免编译错误，当 kBlockM < 64 时，为非 SwapAB 模式使用一个占位类型
using TiledMmaQK_Fallback = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, float, Shape<Int<64>, Int<kBlockN>, Int<kHeadDim>>>(),
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMmaQK_SwapAB = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, float, TileShape_MNK_SwapAB_OP_SELECT>(),
    AtomLayoutQK_SwapAB{}
));

using TiledMmaQK = std::conditional_t<SwapAB, TiledMmaQK_SwapAB, TiledMmaQK_Fallback>;

// 定义 PV 相关的 TileShape
using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
using TileShape_MNK_PV_SwapAB = Shape<Int<kHeadDim>, Int<kBlockM>, Int<kBlockN>>;
using TileShape_MNK_PV_SwapAB_OP_SELECT = Shape<Int<kHeadDim>, Int<TileSize_kBlockM>, Int<kBlockN>>;
using PermutationPV_SwapAB = Tile<Int<kHeadDim>, Int<kBlockM>, Int<kBlockN>>;

// 定义 TiledMmaPV，根据 SwapAB 模式选择不同的配置
// Atom layout for PV is the same as QK
using AtomLayoutPV = Layout<Shape<_1, _1, _1>>;
using AtomLayoutPV_SwapAB = AtomLayoutQK_SwapAB;

// 为了避免编译错误，当 kBlockM < 64 时，为非 SwapAB 模式使用一个占位类型
using TiledMmaPV_Fallback = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, float, Shape<Int<64>, Int<kHeadDim>, Int<kBlockN>>, GMMA::Major::K, GMMA::Major::MN>(),
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMmaPV_SwapAB = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, float, TileShape_MNK_PV_SwapAB_OP_SELECT, GMMA::Major::MN, GMMA::Major::MN>(),
    AtomLayoutPV_SwapAB{},
    PermutationPV_SwapAB{}
));

using TiledMmaPV = std::conditional_t<SwapAB, TiledMmaPV_SwapAB, TiledMmaPV_Fallback>;

// P矩阵拷贝实验核函数
template <typename Element>
__global__ void copy_p_matrix_kernel() {
  // 分配 Shared Memory
  extern __shared__ char smem_buf[];
  Element* smem_p_ptr = reinterpret_cast<Element*>(smem_buf);
  
  // 创建 SmemLayoutP
  auto smem_layout_p = SmemLayoutP{};

  // if (threadIdx.x == 0) {
  //   print("SmemLayoutP: "); print_latex(SmemLayoutP{}); print("\n");
  // }
  
  // 创建 shared memory 张量
  // 需要两个独立的区域：一个存储 thread_id，一个存储 reg_id
  size_t layout_size = size(smem_layout_p);
  IdentityElement* smem_thread_id_ptr = reinterpret_cast<IdentityElement*>(smem_buf);
  IdentityElement* smem_reg_id_ptr = smem_thread_id_ptr + layout_size;
  auto sP_thread_id = make_tensor(make_smem_ptr(smem_p_ptr), smem_layout_p);
  auto sP_reg_id = make_tensor(make_smem_ptr(smem_p_ptr + layout_size), smem_layout_p);
  [[maybe_unused]] auto sP_thread_id_int = make_tensor(make_smem_ptr(smem_thread_id_ptr), smem_layout_p);
  [[maybe_unused]] auto sP_reg_id_int = make_tensor(make_smem_ptr(smem_reg_id_ptr), smem_layout_p);
  
  // 创建 TiledMma
  TiledMmaQK tiled_mma_qk;
  
  // 创建 smem_tiled_copy_P（关键步骤1）
  auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);

  // if (threadIdx.x == 0) {
  //   print("smem_tiled_copy_P: "); print_latex(smem_tiled_copy_P); print("\n");
  // }
  
  // 获取当前线程的拷贝切片（关键步骤2）
  int thread_idx = threadIdx.x;
  auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
  
  // 创建 tPsP - 目标张量分区（关键步骤3）
  // 根据 SwapAB 模式选择不同的构造方式
  // 为两个 shared memory tensor 都创建分区
  auto tPsP_thread_id = [&]() {
    if constexpr (!SwapAB) {
      // Normal mode: keep original tensor construction logic
      return smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP_thread_id));
    } else {
      // SwapAB mode: transpose layout
      auto sP_transposed = make_tensor(
          sP_thread_id.data(),
          cute::make_layout(
              cute::make_shape(get<1>(sP_thread_id.layout().shape()), get<0>(sP_thread_id.layout().shape())),
              cute::make_stride(get<1>(sP_thread_id.layout().stride()), get<0>(sP_thread_id.layout().stride()))));
      return smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP_transposed));
    }
  }();
  
  auto tPsP_reg_id = [&]() {
    if constexpr (!SwapAB) {
      return smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP_reg_id));
    } else {
      auto sP_transposed = make_tensor(
          sP_reg_id.data(),
          cute::make_layout(
              cute::make_shape(get<1>(sP_reg_id.layout().shape()), get<0>(sP_reg_id.layout().shape())),
              cute::make_stride(get<1>(sP_reg_id.layout().stride()), get<0>(sP_reg_id.layout().stride()))));
      return smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP_transposed));
    }
  }();
  
  // 打印线程的布局信息
#ifdef DEBUG_PRINT
  if (thread_idx == 0) {
    printf("=== Configuration ===\n");
    printf("SwapAB: %s\n", SwapAB ? "true" : "false");
    printf("kBlockM: %d, kBlockN: %d, kHeadDim: %d\n", kBlockM, kBlockN, kHeadDim);
    
    printf("\n=== SmemLayoutP ===\n");
    print("SmemLayoutP shape: "); print(smem_layout_p.shape()); print("\n");
    print("SmemLayoutP size: "); print(size(smem_layout_p)); print("\n");
    
    printf("\n=== TiledCopyP ===\n");
    print("smem_tiled_copy_P: "); print(smem_tiled_copy_P); print("\n");

  }
#endif
  
  // 创建 TiledMmaPV 用于 layout 转换
  TiledMmaPV tiled_mma_pv;
  
  // 创建 tSrS - 模拟 Q@K 的输出（accumulator 格式）
  // 在 SwapAB 模式下，这是 K@Q 的结果
  Tensor tSrS = [&]() {
    if constexpr (!SwapAB) {
      return partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
    } else {
      return partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK_SwapAB{}));
    }
  }();
  
  // 初始化 tSrS 的数据（模拟 MMA 的输出）
  for (int i = 0; i < size(tSrS); ++i) {
    tSrS(i) = float(thread_idx * 100 + i);
  }
  

#ifdef DEBUG_PRINT
  if (thread_idx == 0) {
    printf("\n=== tSrS (Q@K accumulator) ===\n");
    print("tSrS shape: "); print(tSrS.shape()); print("\n");
    print("tSrS layout: "); print(tSrS.layout()); print("\n");
    print("tSrS size: "); print(size(tSrS)); print("\n");
    printf("First 8 elements:\n");
    for (int i = 0; i < min(8, int(size(tSrS))); ++i) {
      printf("  tSrS[%d] = %.2f\n", i, float(tSrS(i)));
    }
  }
#endif
  
  // 按照 mainloop 的逻辑转换 layout 和 type
  // Convert layout and type from tSrS to tOrP which will be used in MmaPV
  Tensor tOrP = [&]() {
    if constexpr (kBlockM == 8) {
      Tensor tOrP_acc = make_tensor(tSrS.data(), tSrS.layout());
      Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
      flash::convert_type_out(tOrP_acc, tOrP);
      return tOrP;
    } else {
      Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
      Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
      flash::convert_type_out(tOrP_acc, tOrP);
      return tOrP;
    }
  }();
  

#ifdef DEBUG_PRINT
  if (thread_idx == 0) {
    printf("\n=== tOrP (converted for MmaPV) ===\n");
    print("tOrP shape: "); print(tOrP.shape()); print("\n");
    print("tOrP layout: "); print(tOrP.layout()); print("\n");
    print("tOrP size: "); print(size(tOrP)); print("\n");
    printf("First 8 elements:\n");
    for (int i = 0; i < min(8, int(size(tOrP))); ++i) {
      printf("  tOrP[%d] = %.2f\n", i, float(tOrP(i)));
    }
  }
#endif
  
  // 创建两个独立的 identity tensor 用于理解拷贝对应关系
  // tOrP_identity_thread: 存储 thread_idx
  // tOrP_identity_reg: 存储 register_index
  // 这种方式完全避免编码/解码，没有任何精度问题
  Tensor tOrP_identity_thread_storage = make_tensor_like<IdentityElement>(tOrP);
  Tensor tOrP_identity_reg_storage = make_tensor_like<IdentityElement>(tOrP);
  
  for (int i = 0; i < size(tOrP_identity_thread_storage); ++i) {
    tOrP_identity_thread_storage(i) = static_cast<IdentityElement>(thread_idx);
    tOrP_identity_reg_storage(i) = static_cast<IdentityElement>(i);
  }

  auto tOrP_identity_thread = make_tensor(
      reinterpret_cast<Element*>(tOrP_identity_thread_storage.data()),
      tOrP_identity_thread_storage.layout());
  auto tOrP_identity_reg = make_tensor(
      reinterpret_cast<Element*>(tOrP_identity_reg_storage.data()),
      tOrP_identity_reg_storage.layout());
  
  // 执行拷贝（关键步骤4）
  // 从寄存器拷贝到 shared memory
  // 这是核心的 P 矩阵写入操作
  // 使用两个 identity tensor 来追踪拷贝的对应关系
  cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP_identity_thread), tPsP_thread_id);
  cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP_identity_reg), tPsP_reg_id);
  
  // 同步，确保所有线程完成写入
  __syncthreads();
  
#ifdef DEBUG_PRINT
  // 打印 shared memory 的全局视图（仅线程0）
  if (thread_idx == 0) {
    printf("\n=== Shared Memory 全局视图 (%dx%d P矩阵) ===\n", kBlockM, kBlockN);
    printf("说明：格式为 T{thread_id}:R{reg_id}\n");
    printf("例如：T128:R5 表示来自 Thread 128 的第 5 个寄存器位置\n");
    printf("使用两个独立的 shared memory tensor 存储 thread_id 和 reg_id\n\n");
    
    // 打印更多的行和列来观察拷贝模式
    int rows_to_print = min(16, kBlockM);
    int cols_to_print = min(16, kBlockN);
    
    printf("       ");
    for (int col = 0; col < cols_to_print; ++col) {
      printf("Col%-8d ", col);
    }
    printf("\n");
    
    for (int row = 0; row < rows_to_print; ++row) {
      printf("Row%-3d ", row);
      for (int col = 0; col < cols_to_print; ++col) {
        // 从两个独立的 shared memory tensor 读取
        int thread_id = int(sP_thread_id_int(row, col));
        int reg_idx = int(sP_reg_id_int(row, col));
        printf("T%-3d:R%-3d ", thread_id, reg_idx);
      }
      printf("\n");
    }
    
    // 打印前 200 个 shared memory 位置的详细信息
    printf("\n=== Shared Memory 线性视图（前200个位置）===\n");
    printf("格式：[Index] Offset(bytes) [Bank] <- Value (Thread, Reg)\n\n");
    
    int num_positions = min(200, int(size(smem_layout_p)));
    IdentityElement* thread_id_ptr = smem_thread_id_ptr;
    IdentityElement* reg_id_ptr = smem_reg_id_ptr;
    
    for (int idx = 0; idx < num_positions; ++idx) {
      // 从两个独立的区域读取
      int thread_id = int(thread_id_ptr[idx]);
      int reg_idx = int(reg_id_ptr[idx]);
      
      // 计算 bank ID
      int offset_bytes = idx * int(sizeof(Element));
      int bank_id = (offset_bytes / 4) % 32;
      
      printf("[%3d] %3d bytes [B%2d] <- T%-3d:R%-2d", 
             idx, offset_bytes, bank_id, thread_id, reg_idx);
      
      if ((idx + 1) % 4 == 0) {
        printf("\n");
      } else {
        printf("  |  ");
      }
    }
    if (num_positions % 4 != 0) printf("\n");
    
    // 打印 Shared Memory 地址和 Bank 信息
    printf("\n=== Shared Memory 地址和 Bank 分析 ===\n");
    printf("SmemLayoutP: "); print(smem_layout_p); printf("\n");
    printf("Element size: 2 bytes (FP16)\n");
    printf("每个矩阵 smem size: %lu bytes\n", (unsigned long)(size(smem_layout_p)) * sizeof(Element));
    printf("Total smem size: %lu bytes (thread_id + reg_id 两个矩阵)\n\n", 2 * (unsigned long)(size(smem_layout_p)) * sizeof(Element));
    
    // 分析真实的 Bank Conflict - 通过线性地址遍历
    printf("\n=== Bank Conflict 分析（基于物理地址的 Warp-Level Copy）===\n");
    printf("通过线性遍历 shared memory 来获取真实的物理地址和 bank 分布\n\n");
    
    // 第一步：建立物理地址 -> (thread, reg) 的映射
    struct AddrInfo {
      int thread_id;
      int reg_idx;
      int bank_id;
      bool valid;
    };
    
    int total_elems = int(size(smem_layout_p));
    AddrInfo* addr_map = new AddrInfo[total_elems];
    
    // 遍历物理地址，建立映射
    // thread_id_ptr 和 reg_id_ptr 已经在上面声明过了，直接使用
    
    for (int idx = 0; idx < total_elems; ++idx) {
      // 从两个独立的区域读取
      int thread_id = int(thread_id_ptr[idx]);
      int reg_idx = int(reg_id_ptr[idx]);
      int offset_bytes = idx * int(sizeof(Element));
      int bank_id = (offset_bytes / 4) % 32;
      
      addr_map[idx].thread_id = thread_id;
      addr_map[idx].reg_idx = reg_idx;
      addr_map[idx].bank_id = bank_id;
      addr_map[idx].valid = (thread_id >= 0 && thread_id < 128 && reg_idx >= 0 && reg_idx < 64);  // 合理范围：128个线程, 64个寄存器
    }
    
    // 第二步：按寄存器索引分组，分析每个 warp 的 bank conflict
    // 自动检测寄存器数量
    int max_reg_idx = 0;
    for (int idx = 0; idx < total_elems; ++idx) {
      if (addr_map[idx].valid && addr_map[idx].reg_idx > max_reg_idx) {
        max_reg_idx = addr_map[idx].reg_idx;
      }
    }
    int num_regs_per_thread = max_reg_idx + 1;
    
    int num_warps_to_check = 4;  // 检查所有4个warp (128线程 = 4个warp)
    int num_regs_to_check = num_regs_per_thread;  // 检查所有寄存器
    
    printf("检测到每个线程有 %d 个寄存器\n", num_regs_per_thread);
    printf("将检查 %d 个 warp 的所有 %d 个寄存器拷贝操作\n\n", num_warps_to_check, num_regs_to_check);
    
    for (int warp_id = 0; warp_id < num_warps_to_check; ++warp_id) {
      printf("=== Warp %d (Threads %d-%d) ===\n", warp_id, warp_id * 32, warp_id * 32 + 31);
      
      for (int reg_idx = 0; reg_idx < num_regs_to_check; ++reg_idx) {
        printf("\nCopy 操作 %d (每个线程拷贝寄存器索引 %d):\n", reg_idx, reg_idx);
        
        int bank_histogram[32] = {0};
        int lane_to_bank[32];
        int lane_to_addr[32];
        
        // 初始化
        for (int i = 0; i < 32; ++i) {
          lane_to_bank[i] = -1;
          lane_to_addr[i] = -1;
        }
        
        // 遍历所有地址，找到这个 warp 中每个线程的该寄存器写入位置
        for (int idx = 0; idx < total_elems; ++idx) {
          if (!addr_map[idx].valid) continue;
          
          int tid = addr_map[idx].thread_id;
          int rid = addr_map[idx].reg_idx;
          
          // 检查是否属于当前 warp 和当前寄存器
          if (tid >= warp_id * 32 && tid < (warp_id + 1) * 32 && rid == reg_idx) {
            int lane = tid - warp_id * 32;
            if (lane_to_bank[lane] == -1) {
              lane_to_bank[lane] = addr_map[idx].bank_id;
              lane_to_addr[lane] = idx;
              bank_histogram[addr_map[idx].bank_id]++;
            } else {
              printf("\n  T%-3d->Addr[%4d]->B%-2d (Conflict with T%-3d->Addr[%4d]->B%-2d)\n", 
                     tid, idx, addr_map[idx].bank_id, tid, lane_to_addr[lane], lane_to_bank[lane]);
              bank_histogram[addr_map[idx].bank_id]++;
            }
          }
        }
        
        // 打印前16个线程的信息
        for (int lane = 0; lane < 32; ++lane) {
          int thread_id = warp_id * 32 + lane;
          if (lane_to_addr[lane] >= 0) {
            printf("  T%-3d->Addr[%4d]->B%-2d  ", 
                   thread_id, lane_to_addr[lane], lane_to_bank[lane]);
            if ((lane + 1) % 4 == 0) printf("\n");
          }
        }
        if (16 % 4 != 0) printf("\n");
        
        // 分析 bank conflict
        int max_conflicts = 0;
        int conflict_banks = 0;
        int active_threads = 0;
        
        for (int i = 0; i < 32; ++i) {
          if (bank_histogram[i] > max_conflicts) {
            max_conflicts = bank_histogram[i];
          }
          if (bank_histogram[i] > 1) {
            conflict_banks++;
          }
        }
        
        for (int lane = 0; lane < 32; ++lane) {
          if (lane_to_addr[lane] >= 0) active_threads++;
        }
        
        printf("\n  活跃线程数: %d/32\n", active_threads);
        printf("  Bank 分布统计: ");
        for (int i = 0; i < 32; ++i) {
          if (bank_histogram[i] > 0) {
            printf("B%d:%d次 ", i, bank_histogram[i]);
          }
        }
        printf("\n");
        
        if (max_conflicts > 1) {
          printf("  ⚠️  检测到 Bank Conflict! 最大冲突度: %d-way，%d 个 bank 有冲突\n", 
                 max_conflicts, conflict_banks);
        } else if (active_threads > 0) {
          printf("  ✓  无 Bank Conflict\n");
        } else {
          printf("  (该 warp 无活跃线程)\n");
        }
      }  // end of reg_idx loop
      printf("\n");
    }  // end of warp_id loop
    
    delete[] addr_map;
  }
#endif  // DEBUG_PRINT
}

int main() {
  std::cout << "=== P矩阵拷贝实验 (SwapAB模式) ===" << std::endl;
  std::cout << "模拟 mainloop_fwd_sm90_tma_gmma_ws.hpp 中的 P 拷贝逻辑\n" << std::endl;
  
  std::cout << "配置参数:" << std::endl;
  std::cout << "  SwapAB = " << (SwapAB ? "true" : "false") << std::endl;
  std::cout << "  kBlockM = " << kBlockM << std::endl;
  std::cout << "  kBlockN = " << kBlockN << std::endl;
  std::cout << "  kHeadDim = " << kHeadDim << std::endl;
  std::cout << "  TileSize_kBlockM = " << TileSize_kBlockM << std::endl;
  std::cout << "\n启动 kernel..." << std::endl;
  
  // 计算需要的 shared memory 大小
  // 需要两倍空间：一个存储 thread_id，一个存储 reg_id
  size_t smem_size = 2 * kBlockM * kBlockN * sizeof(Element);
  std::cout << "Shared Memory 大小: " << smem_size << " bytes (两个 " << kBlockM << "x" << kBlockN << " 矩阵)" << std::endl;
  
  // 使用 128 个线程（1个 warp group）
  int num_threads = 128;
  
  // 启动 kernel
  copy_p_matrix_kernel<Element><<<1, num_threads, smem_size>>>();
  
  // 同步并检查错误
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA 错误: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  
  return 0;
}
// 编译命令：
// nvcc -std=c++17 -O3 -arch=sm_90 -I magi_attention/csrc/cutlass/include -I magi_attention/csrc/common --expt-relaxed-constexpr --expt-extended-lambda main.cu -o cute_test
// ./cute_test
// ./cute_test