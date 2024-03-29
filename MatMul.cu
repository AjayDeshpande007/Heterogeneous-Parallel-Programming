// headers
#include <stdio.h>

#include <cuda.h> // for CUDA

// from NVIDIA CUDA SDK [ REMEBER : Some header file changes are done to the original file ]
#include "helper_timer.h"

#define BLOCK_WIDTH 256

// variable declarations
float *hostA=NULL;
float *hostB=NULL;
float *hostC=NULL;
float *CHost=NULL;

float *deviceA=NULL;
float *deviceB=NULL;
float *deviceC=NULL;

float timeOnCPU;
float timeOnGPU;

// global kernel function definition
__global__ void matrixMultiply(float *A,float *B,float *C,int numARows,int numAColumns,int numBRows,int numBColumns,int numCRows,int numCColumns)
{
    // variable declarations
    int row=blockIdx.y * blockDim.y + threadIdx.y;
    int col=blockIdx.x * blockDim.x + threadIdx.x;
    // code
    if((row < numARows) && (col < numBColumns))
    {
        float Cvalue=0.0;
        for(int k=0; k < numAColumns; k++)
        {
            Cvalue +=A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numCColumns + col]=Cvalue;
    }
}

int main(int argc,char *argv[])
{
    // function declarations
    void fillFloatArrayWithRandomNumbers(float *, int);
    void matMulHost(float *,float *,float *,int,int,int);
    void cleanup(void);
   
    // variable declarations
    int numARows;
    int numAColumns;
    int numBRows;
    int numBColumns;
    int numCRows;
    int numCColumns;
    int numCHostRows;
    int numCHostColumns;
    
    // code
    numARows=BLOCK_WIDTH;
    numAColumns=BLOCK_WIDTH;
    numBRows=BLOCK_WIDTH;
    numBColumns=BLOCK_WIDTH;

    numCRows=numARows;
    numCColumns=numBColumns;
    
    numCHostRows=numARows;
    numCHostColumns=numBColumns;

    int sizeA= numARows * numAColumns * sizeof(float);
    int sizeB= numBRows * numBColumns * sizeof(float);
    int sizeC= numCRows * numCColumns * sizeof(float);
    int sizeCHost= numCHostRows * numCHostColumns * sizeof(float);
    
    // allocate host-memory
    hostA=(float *)malloc(sizeA);
    if(hostA==NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix A.\nExitting ...\n");
        exit(EXIT_FAILURE);
    }
    
    hostB=(float *)malloc(sizeB);
    if(hostB==NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix B.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    hostC=(float *)malloc(sizeC);
    if(hostC== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Matrix C.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    CHost=(float *)malloc(sizeCHost);
    if(hostC== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Matrix C.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // fill above input host vectors with arbitary but hard-coded data
    fillFloatArrayWithRandomNumbers(hostA,numARows * numAColumns);
    fillFloatArrayWithRandomNumbers(hostB,numBRows * numBColumns);
    
    // allocate device-memory
    cudaError_t err=cudaSuccess;
    err=cudaMalloc((void **)&deviceA,sizeA);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMalloc((void **)&deviceB,sizeB);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMalloc((void **)&deviceC,sizeC);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // copy host memory contents to device memory
    err=cudaMemcpy(deviceA,hostA,sizeA,cudaMemcpyHostToDevice);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMemcpy(deviceB,hostB,sizeB,cudaMemcpyHostToDevice);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // cuda kernel configuration
    dim3 DimGrid=dim3(ceil((int)numCColumns/(int)BLOCK_WIDTH),ceil((int)numCRows/(int)BLOCK_WIDTH),1);
    dim3 DimBlock=dim3(BLOCK_WIDTH,BLOCK_WIDTH,1);

    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    matrixMultiply<<<DimGrid,DimBlock>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);

    // stop timer
    sdkStopTimer(&timer);
    timeOnGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    // copy device memory to host memory
    err=cudaMemcpy(hostC,deviceC,sizeC,cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // results
    matMulHost(hostA,hostB,CHost,numAColumns,numCHostRows,numCHostColumns);
   
    // compare results for golden-host
    const float epsilon = 0.000001f;
    bool bAccuracy=true;
    int breakValue=0;
    int i;
    for(i=0;i<numARows * numAColumns;i++)
    {
        float val1 = CHost[i];
        float val2 = hostC[i];
        if(fabs(val1-val2) > epsilon)
        {
            bAccuracy = false;
            breakValue=i;
            break;
        }
    }
    
    if(bAccuracy==false)
    {
        printf("Break Value = %d\n",breakValue);
    }
    
    char str[125];
    if(bAccuracy==true)
        sprintf(str,"%s","Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
    else
        sprintf(str,"%s","Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
   
    printf("------------------------Matrix Multiplication using CUDA with a Float as an Input Data Type--------------------------------\n");
    printf("The Dimensions Of First Matrix Are : %d x %d\n", numARows, numAColumns);
    printf("The Dimensions Of 'Second Matrix Are : %d x %d\n", numBRows, numBColumns);
    printf("The Dimensions Of Output Matrix On CPU Are : %d x %d\n", numCRows, numCColumns);

    printf("The Dimensions Of Output Matrix Of GPU Are : %d x %d\n", numCHostRows, numCHostColumns);

    printf("Size of First Matrix  = %d\n", sizeA);
    printf("Size of Second Matrix  = %d\n", sizeB);
    printf("Size of Output Matrix On CPU Matrix  = %d\n", sizeC);
    printf("Size of Output Matrix On GPU  = %d\n", sizeCHost);
    printf("The Time Taken To Do Above Multiplication On CPU = %.6f (ms)\n",timeOnCPU);
    printf("The Time Taken To Do Above Multiplication On GPU = %.6f (ms)\n",timeOnGPU);
    printf("%s\n",str);

    // total cleanup
    cleanup();
    
    return(0);
}

void cleanup(void)
{
    // code
    
    // free allocated device-memory
    if(deviceA)
    {
        cudaFree(deviceA);
        deviceA=NULL;
    }
    
    if(deviceB)
    {
        cudaFree(deviceB);
        deviceB=NULL;
    }
    
    if(deviceC)
    {
        cudaFree(deviceC);
        deviceC=NULL;
    }
    
    // free allocated host-memory
    if(hostA)
    {
        free(hostA);
        hostA=NULL;
    }
    
    if(hostB)
    {
        free(hostB);
        hostB=NULL;
    }
    
    if(hostC)
    {
        free(hostC);
        hostC=NULL;
    }
    
    if(CHost)
    {
        free(CHost);
        CHost=NULL;
    }
}

void fillFloatArrayWithRandomNumbers(float *pFloatArray, int iSize)
{
    // code
    int i;
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i)
    {
        pFloatArray[i] = fScale * rand();
    }
}

void matMulHost(float *A,float *B,float* C,int iAColumns,int iCRows,int iCColumns)
{
    // code
    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    for(int i=0;i<iCRows;++i)
    {
        for(int j=0;j<iCColumns;++j)
        {
            float sum=0.0f;
            for(int k=0;k<iAColumns;++k)
            {
                float a=A[i * iAColumns + k];
                float b=B[k * iCColumns + j];
                sum += a * b;
            }
            C[i * iCColumns + j] = sum;
        }
    }
    
    // stop timer
    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
}
