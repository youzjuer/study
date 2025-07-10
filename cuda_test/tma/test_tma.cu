#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// nvcc -o tma tma.cu -O2 -arch=sm_90a -std=c++17 && ./tma

__global__ void kernel(half* ptr, int elts)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
 extern __shared__ __align__(16) half smem[];
  
  ////////////////// global mem -> shared mem //////////////////
  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                      // a)
    cde::fence_proxy_async_shared_cta();         // b)
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    cuda::memcpy_async(
        smem, 
        ptr,
        cuda::aligned_size_t<16>(sizeof(half)*elts),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < elts; i += blockDim.x) {
    smem[i] += 1;
  }
  
  ////////////////// shared mem -> global mem //////////////////
  // 5. Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();   // b)
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  if(threadIdx.x == 0) {
    printf("\ndata on device:\n");
    for(int i = 0; i < elts; i ++) {
        printf("%.2lf ", float(smem[i]));
    }
  }

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_shared_to_global(
            ptr, smem, sizeof(half)*elts);
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }
}

int main() {
    int warps_per_block{4}, threads_per_warps{32}, elts_per_threads{8};
    int elts = warps_per_block*threads_per_warps*elts_per_threads;
    half *h_ptr;
    h_ptr = (half*)malloc(sizeof(half)*elts);

    auto fn = [](half *ptr, int elts) {
        std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<float> dis{};
        for(int i = 0; i < elts; i ++) {
            ptr[i] = (half)dis(gen);
        }
    };
    fn(h_ptr, elts);

    printf("\ndata on host:\n");
    for(int i = 0; i < elts; i ++) {
        printf("%.2lf ", float(h_ptr[i]));
    }

    half *d_ptr;
    cudaMalloc(&d_ptr, sizeof(half)*elts);
    cudaMemcpy(d_ptr, h_ptr, sizeof(half)*elts, cudaMemcpyHostToDevice);

    int block_size = warps_per_block * threads_per_warps;
    size_t smems_size = sizeof(half)*elts;
    kernel<<<1, block_size, smems_size>>>(d_ptr, elts);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ptr, d_ptr, sizeof(half)*elts, cudaMemcpyDeviceToHost);
    printf("\ndata on host after add one:\n");
    for(int i = 0; i < elts; i ++) {
        printf("%.2lf ", float(h_ptr[i]));
    }

    return 0;
}