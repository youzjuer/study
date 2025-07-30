#include<cuda.h>
#include<cuda_runtime.h>
#include<torch/extension.h>

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 1024
#endif
#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif
constexpr int URF{UNROLL_FACTOR};



template <typename scalar_t>
__global__ void softmax_kernel(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    float maxval = a[row*w];
    for (int i = 1; i<w; i++)
    {
      maxval = fmaxf(maxval, a[row*w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }
    b[row*w + col] = __expf(a[row*w + col]-maxval)/(divisor);
  }
}
template<typename scalar_t>
__global__ void  softmax_kernel2(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h){
    //[0~h-1]:split block
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    //[0~BLOCK_DIM_Y-1]:split thread
    int ty = threadIdx.y;
    __shared__ float reduction[BLOCK_DIM_Y];
    if(row < h){
      float maxval = 0;
      for(int i = ty * BLOCK_DIM_Y; i < min(w, (ty+1) * BLOCK_DIM_Y); i++){
          maxval = fmaxf(maxval,a[row * w + i]);
      }
      reduction[ty] = maxval;
      for(int stride = BLOCK_DIM_Y / 2; stride >= 1; stride = stride / 2){
          __syncthreads();
          if(ty < stride){
              reduction[ty] = fmaxf(reduction[ty],reduction[ty + stride]);
          }
      }
      __syncthreads();
      // row max value
      maxval = reduction[0];
      float divisor = 0.0f;
      for(int i = ty *BLOCK_DIM_Y; i < min(w, (ty + 1) * BLOCK_DIM_Y); i++){
          divisor += __expf(a[row * w + i] - maxval);
      }
      reduction[ty] = divisor;
      for(int stride = BLOCK_DIM_Y / 2; stride >= 1; stride = stride / 2){
          __syncthreads();
          if(ty < stride){
              reduction[ty] += reduction[ty + stride];
          }
      }
      __syncthreads();
      divisor = reduction[0];
      for(int i = ty; i < w; i = i + BLOCK_DIM_Y){
        b[row * w + i ] = __expf(a[row * w + i] - maxval) / divisor;
      }
    }
}
template <typename scalar_t>
__global__ void softmax_kernel3(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float reduction[BLOCK_DIM_Y]; 
  if (row < h)
  {
    float maxval = 0;
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      maxval = fmaxf(maxval, a[row*w + i]);
    }

    reduction[ty] = maxval;
    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = fmaxf(reduction[ty], reduction[ty+stride]);
      }
    }

    __syncthreads();
    maxval = reduction[0];

    float divisor = 0.f;
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }
    reduction[ty] = divisor;
    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = reduction[ty] + reduction[ty+stride];
      }
    }
    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}
template <typename scalar_t>
__global__ void softmax_kernel4(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty / 32;


  __shared__ float reduction[BLOCK_DIM_Y / 32];
  if(row < h){
    float maxval = 0;
    // find maxval of every thread's processing data
    for(int i = ty; i < w; i += BLOCK_DIM_Y){
      maxval = fmaxf(maxval, a[row *w + i]);
    }
    for(int mask = 16;  mask > 0 ; mask /=2){
      maxval = fmaxf(maxval,__shfl_xor_sync(0xffffffff,maxval,mask,32));
    }
    if( ty % 32 == 0){
      reduction[warp_id] = maxval;
    }
    __syncthreads();
    if(warp_id == 0){
      for(int mask = 16; mask > 0; mask /= 2){
        maxval = ty < BLOCK_DIM_Y / 32 ? reduction[ty]:0;
        maxval = fmaxf(maxval,__shfl_xor_sync(0xffffffff,maxval,mask,32));
      }
    }
    if(ty == 0){
      reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    float divisor = 0.f;
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }
    for (int mask = 16; mask>0; mask/=2)
    {
      divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}
// template <typename scalar_t>
// __global__ void softmax_kernel5(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h){
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int ty = threadIdx.y;
//     __shared__ float reduction[BLOCK_DIM_Y/2];
    
//     if(row < h){
//       float maxval = 0;
//       for(int i = ty; i < w / 4; i += BLOCK_DIM_Y){
//         float4 val = reinterpret_cast<float4*>(&a[row * w + i * 4])[0];
//         maxval = fmaxf(maxval,val.x);
//         maxval = fmaxf(maxval,val.y);
//         maxval = fmaxf(maxval,val.z);
//         maxval = fmaxf(maxval,val.w);
//       }
//       if(ty >= BLOCK_DIM_Y / 2){
//         reduction[ty - BLOCK_DIM_Y/2] = maxval;
//       }
//       for(int stride = BLOCK_DIM_Y / 2; stride > 0; stride /= 2){
//         __syncthreads();
//         if(ty < stride){
//           maxval = fmaxf(maxval, reduction[ty]);
//           if(ty >= stride / 2){
//             reduction[ty - stride / 2] = maxval; 
//           }
//         }
//       }
//       __syncthreads();
//       maxval = reduction[0];
//       float sum = 0.0f;
//       for(int i = ty; i < w / 4; i += BLOCK_DIM_Y){
//         float4 val = reinterpret_cast<float4*>(&a[row * w + i * 4])[0];
//         sum += __expf(val.x - maxval); 
//         sum += __expf(val.y - maxval); 
//         sum += __expf(val.z - maxval); 
//         sum += __expf(val.w - maxval); 
//       }
//       if(ty >= BLOCK_DIM_Y / 2){
//         reduction[ty - BLOCK_DIM_Y / 2] = sum;
//       }
//       for(int stride = BLOCK_DIM_Y / 2 ;  stride > 0; stride /= stride){
//         __syncthreads();
//         if(ty  < stride){
//           sum += reduction[ty];
        
//         if(ty >= stride / 2){
//           reduction[ty - stride / 2] = sum;
//         }
//       }
//       }
//       __syncthreads();
//       sum = reduction[0];
//       for(int i = 0; i < w /4; i += BLOCK_DIM_Y){
//         float4 val = reinterpret_cast<float4*>(&a[row * w + i * 4])[0];
//         val.x = __expf(val.x - maxval) / sum;
//         val.y = __expf(val.y - maxval) / sum;
//         val.z = __expf(val.z - maxval) / sum;
//         val.w = __expf(val.w - maxval) / sum;
//         reinterpret_cast<float4*>(&b[row * w + i * 4])[0] = val;
//       }
//     }
// }
template <typename scalar_t>
__global__ void softmax_kernel5(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float reduction[BLOCK_DIM_Y/2]; 
  if (row < h)
  {
    float maxval = 0;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      maxval = fmaxf(maxval, val.x);
      maxval = fmaxf(maxval, val.y);
      maxval = fmaxf(maxval, val.z);
      maxval = fmaxf(maxval, val.w);
    }

    if (ty >= BLOCK_DIM_Y/2)
    {
      reduction[ty - BLOCK_DIM_Y/2] = maxval;
    }
    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        maxval = fmaxf(maxval, reduction[ty]);
        if (ty >= stride/2)
        {
          reduction[ty - stride/2] = maxval;
        }
      }
    }

    __syncthreads();
    maxval = reduction[0];

    float divisor = 0.f;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      divisor += __expf(val.x - maxval);
      divisor += __expf(val.y - maxval);
      divisor += __expf(val.z - maxval);
      divisor += __expf(val.w - maxval);
    }

    if (ty >= BLOCK_DIM_Y/2)
    {
      reduction[ty - BLOCK_DIM_Y/2] = divisor;
    }

    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        divisor = divisor + reduction[ty];
        if (ty >= stride/2)
        {
          reduction[ty - stride/2] = divisor;
        }
      }
    }
    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        val.x = __expf(val.x-maxval)/divisor;
        val.y = __expf(val.y-maxval)/divisor;
        val.z = __expf(val.z-maxval)/divisor;
        val.w = __expf(val.w-maxval)/divisor;
        reinterpret_cast<float4*>(&b[row*w + i*4])[0] = val;
    }
  }
}
template <typename scalar_t>
__global__ void softmax_kernel6(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h){
  // int row = blockIdx.x * blockDim.x + threadIdx.x ;
  int row = blockIdx.x;
  int ty = threadIdx.y;
  int warp_id = threadIdx.y / 32;

  //BLOCK_DIM_Y/32 = warp的数量
  __shared__ float reduction[BLOCK_DIM_Y/32];
  if(row < h){
    float maxval = 0.0f;
    // 画个图就都懂了,every maxval represents max value of every thread
    #pragma unroll URF
    for(int i = ty; i < w / 4; i += BLOCK_DIM_Y){
      float4 val = reinterpret_cast<float4*>(&a[row * w + i * 4])[0];
      maxval = fmaxf(maxval,val.x);
      maxval = fmaxf(maxval,val.y);
      maxval = fmaxf(maxval,val.z);
      maxval = fmaxf(maxval,val.w);
    }
    // warp reduce
    for(int mask = 16; mask > 0 ; mask /= 2){
      maxval = fmaxf(maxval,__shfl_down_sync(0xffffffff,maxval,mask,32));
    }
    // thread block reduce,represent every warp maxval
    if(ty % 32 == 0){
      reduction[ty / 32] = maxval;
    }
    __syncthreads();
    // calculate diff warp max value
    if(warp_id == 0){
      maxval = ty < BLOCK_DIM_Y / 32? reduction[ty]:0;
      for(int mask = 16 ; mask >=1; mask /=2){
        maxval = fmaxf(maxval,__shfl_xor_sync(0xffffffff,maxval,mask,32));
      }
    }
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    float divisor = 0.0f;
    for(int i = ty; i < w / 4; i += BLOCK_DIM_Y){
      float4 val = reinterpret_cast<float4*>(&a[row * w + i * 4])[0];
      divisor += __expf(val.x - maxval);
      divisor += __expf(val.y - maxval);
      divisor += __expf(val.z - maxval);
      divisor += __expf(val.w - maxval);
    }
    // warp reduce
    for(int mask = 16; mask > 0 ; mask /= 2){
      divisor += __shfl_xor_sync(0xffffffff,divisor,mask,32);
    }
    // thread block reduce
    if(ty % 32 == 0){
      reduction[ty / 32] = divisor;
    }
    __syncthreads();
    // 一般一个block最多1024个thread,因此下面的代码可以成立
    if(warp_id == 0){
      divisor = ty < BLOCK_DIM_Y / 32? reduction[ty]:0;
      for(int mask = 16 ; mask >=1; mask /=2){
        divisor += __shfl_xor_sync(0xffffffff,divisor,mask,32);
      }
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }
    __syncthreads();
    divisor = reduction[0];
    // output
    for(int i = ty; i < w / 4; i += BLOCK_DIM_Y){
      float4 val = reinterpret_cast<float4*>(&a[row * w + i * 4])[0];
      val.x = __expf(val.x - maxval)/divisor;
      val.y = __expf(val.y - maxval)/divisor;
      val.z = __expf(val.z - maxval)/divisor;
      val.w = __expf(val.w - maxval)/divisor;
      reinterpret_cast<float4*>(&b[row * w + i * 4])[0] = val;
    }
  }
}
template <typename scalar_t>
__global__ void softmax_kernel7(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  __shared__ float reduction[BLOCK_DIM_Y/32]; 
  if (row < h)
  {
    float maxval = 0;
#pragma unroll URF
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        maxval = fmaxf(maxval, val.x);
        maxval = fmaxf(maxval, val.y);
        maxval = fmaxf(maxval, val.z);
        maxval = fmaxf(maxval, val.w);
    }
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 16, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 8, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 4, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 2, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 1, 32));

    if (ty%32 == 0)
    {
      reduction[warp_id] = maxval;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 16, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 8, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 4, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 2, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 1, 32));
    }
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    float divisor = 0.f;
#pragma unroll URF
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        divisor += __expf(val.x - maxval);
        divisor += __expf(val.y - maxval);
        divisor += __expf(val.z - maxval);
        divisor += __expf(val.w - maxval);
    }

    divisor += __shfl_down_sync(0xffffffff, divisor, 16, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 8, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 4, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 2, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 1, 32);

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        divisor += __shfl_down_sync(0xffffffff, divisor, 16, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 8, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 4, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 2, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 1, 32);
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

#pragma unroll URF
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        val.x = __expf(val.x-maxval)/divisor;
        val.y = __expf(val.y-maxval)/divisor;
        val.z = __expf(val.z-maxval)/divisor;
        val.w = __expf(val.w-maxval)/divisor;
        reinterpret_cast<float4*>(&b[row*w + i*4])[0] = val;
    }
  }
}
torch::Tensor softmax_cu(torch::Tensor x){
    auto out = torch::empty_like(x);
    int h = x.size(0);
    int w = x.size(1);

    dim3 block_size = dim3(1,BLOCK_DIM_Y,1);
    dim3 grid_size = dim3(h,1,1);

    #if SOFTMAX_VARIANT == 1
        block_size = dim3(32, 32, 1);
        grid_size = dim3(w/32, h/32, 1);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel<scalar_t><<<grid_size, block_size>>>
            (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, h);
        }));
    #endif
    #if SOFTMAX_VARIANT == 2
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel2<scalar_t><<<grid_size, block_size>>>
                (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, h);
        }));
    #endif

    #if SOFTMAX_VARIANT == 3
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
        softmax_kernel3<scalar_t><<<grid_size, block_size>>>
          (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, h);
        }));
    #endif
    #if SOFTMAX_VARIANT == 4
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel4<scalar_t><<<grid_size, block_size>>>
              (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, h);
            }));
    #endif
    #if SOFTMAX_VARIANT == 5
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel5<scalar_t><<<grid_size, block_size>>>
              (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, h);
            }));
    #endif
    #if SOFTMAX_VARIANT == 6
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel6<scalar_t><<<grid_size, block_size>>>
              (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, h);
            }));
    #endif
    #if SOFTMAX_VARIANT == 7 
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_cuda", ([&] {
            softmax_kernel7<scalar_t><<<grid_size, block_size>>>
              (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, h);
            }));
    #endif

    return out;
}