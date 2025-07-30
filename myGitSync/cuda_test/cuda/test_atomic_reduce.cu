#include<iostream>
#include<algorithm>
#include<cuda_runtime.h>
#include<device_launch_paramters.h>


template<int vec_size, typename scalar_t>
void test_reduce_gpu(size_t n){

}





int main(){
    std::cout << "1GB reduce test...."<<std::endl;
    std::cout << "float4"<< std::endl;
    test_reduce_gpu<4,float>(1024*1024*256);
}