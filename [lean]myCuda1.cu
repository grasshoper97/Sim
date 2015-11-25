//#include "cuda_runtime.h"
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include "device_launch_parameters.h"
#include<stdio.h>
//#include<stdlib.h>
//#include<string.h>
//#include<math.h>
//#include<cutil.h>
 

#include<iostream>
#define NUM 2048

//////////////////////////////////////////////////////////////////
__global__ void 
    caculate(unsigned char * g_x)
{
    __shared__ char s[512];
	const unsigned int bid=blockIdx.x*blockDim.x+threadIdx.x;
    if(bid>=NUM) return;
    s[bid]=g_x[bid];
     s[bid]*=2;  
     g_x[bid]=s[bid];

}
 
///////////////////////////////////////////////////////////////
    int main(int argc,char**argv)
{


     //重定向到文件
    //freopen("1.txt", "w", stdout);
    int SIZE=sizeof(unsigned char);
    //----------------------------------------
    unsigned char *h_x=(unsigned char*)malloc(SIZE*NUM);
    for(int i=0;i<NUM;i++)
        h_x[i]=100;  
    //---------------------------
    unsigned char *d_x;    
    cudaMalloc((void**)&d_x,SIZE*NUM);

    //输入数组从内存拷贝到显存
    cudaMemcpy(d_x,h_x,SIZE*NUM,cudaMemcpyHostToDevice);

     //调用核心函数
    dim3 grid;
    dim3 block;
    block.x=512;
    grid.x=(NUM+block.x-1)/block.x;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double sum=0;

    for(int i=0;i<1;i++){
    cudaEventRecord(start, 0);
    float runTime;
    //====================================
    caculate<<<grid,block>>>(d_x);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runTime, start, stop);
    printf("kernel error no =[%d]",cudaGetLastError()); 
    printf("time= %f\n ",runTime);
    sum+=runTime;

    }
    printf("aver time= %f\n ",sum);

    //两个同步语句cudaThreadSynchronize、cudaDeviceSynchronize必须有一个才能让nsight中显示内核函数。
    //cudaThreadSynchronize(); 
    //=====================================



    //CUT_CHECK_ERROR("Kernel execution failed");
    //输出数组从显存拷贝到内存
    cudaMemcpy(h_x,d_x,SIZE*NUM,cudaMemcpyDeviceToHost);
    //在主机端打印
    //for(int i=0;i<NUM;i++)
    printf("h_x[0]=[%c]\n",h_x[0]); 
    printf("h_x[num-1]=[%c]\n",h_x[NUM-1]); 

    //释放内存、显存    
    free(h_x);
    cudaFree(d_x);


    printf("press enter to quit:");
    getchar();
}


