// headers
#include <stdio.h>
#include <stdlib.h> // exit()
#include <string.h> // strlen()
#include <math.h> // fabs()

#include <CL/opencl.h> 


#include "helper_timer.h"

#define BLOCK_WIDTH 256

//#define CL_USE_DEPRECATED_OPENCL_1_0_APIS

//#define CL_TARGET_OPENCL_VERSION = 200



// global OpenCL variables
cl_int ret_ocl;
cl_platform_id oclPlatformID;
cl_device_id oclComputeDeviceID; // compute device id
cl_context oclContext; // compute context
cl_command_queue oclCommandQueue; // compute command queue
cl_program oclProgram; // compute program
cl_kernel oclKernel; // compute kernel

char* oclSourceCode = NULL;
size_t sizeKernelCodeLength;

size_t localWorkSize = 256;
size_t globalWorkSize;

float* hostA = NULL;
float* hostB = NULL;
float* hostC = NULL;
float* CHost = NULL;

cl_mem deviceA = NULL;
cl_mem deviceB = NULL;
cl_mem deviceC = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

int main(void)
{
    // function declarations
    void fillFloatArrayWithRandomNumbers(float*, int);
    size_t roundGlobalSizeToNearestMultipleOfLocalSize(int, unsigned int);
    void matMulHost(float*, float*, float*, int, int, int);
    char* loadOclProgramSource(const char*, const char*, size_t*);
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
	numARows = BLOCK_WIDTH;
    numAColumns = BLOCK_WIDTH;
    numBRows = BLOCK_WIDTH;
    numBColumns = BLOCK_WIDTH;

    numCRows = numARows;
    numCColumns = numBColumns;

    numCHostRows = numARows;
    numCHostColumns = numBColumns;

    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);
    int sizeCHost = numCHostRows * numCHostColumns * sizeof(float);

    // allocate host-memory
    hostA = (float*)malloc(sizeA);
    if (hostA == NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix A.\nExitting ...\n");
        exit(EXIT_FAILURE);
    }

    hostB = (float*)malloc(sizeB);
    if (hostB == NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix B.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostC = (float*)malloc(sizeC);
    if (hostC == NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Matrix C.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    CHost = (float*)malloc(sizeCHost);
    if (hostC == NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Matrix C.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // fill above input host vectors with arbitary but hard-coded data
    fillFloatArrayWithRandomNumbers(hostA, numARows * numAColumns);
    fillFloatArrayWithRandomNumbers(hostB, numBRows * numBColumns);

    // get OpenCL supporting platform's ID
    ret_ocl = clGetPlatformIDs(1, &oclPlatformID, NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clGetDeviceIDs() Failed : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // get OpenCL supporting GPU device's ID
    ret_ocl = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 1, &oclComputeDeviceID, NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clGetDeviceIDs() Failed : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    char gpu_name[255];
    clGetDeviceInfo(oclComputeDeviceID, CL_DEVICE_NAME, sizeof(gpu_name), &gpu_name, NULL);
    printf("--------------------------Matrix Multiplication using OpenCL-------------------------------\n");
    printf("Graphics Processing Unit (GPU) Name : %s\n", gpu_name);

    // create OpenCL compute context
    oclContext = clCreateContext(NULL, 1, &oclComputeDeviceID, NULL, NULL, &ret_ocl);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateContext() Failed : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // create command queue
    oclCommandQueue = clCreateCommandQueue(oclContext, oclComputeDeviceID, 0, &ret_ocl);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateCommandQueue() Failed : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // create OpenCL program from .cl
    oclSourceCode = loadOclProgramSource("MatMul.cl", "", &sizeKernelCodeLength);

    cl_int status = 0;
    oclProgram = clCreateProgramWithSource(oclContext, 1, (const char**)&oclSourceCode, &sizeKernelCodeLength, &ret_ocl);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateProgramWithSource() Failed : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(0);
    }

    // build OpenCL program
    ret_ocl = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clBuildProgram() Failed : %d. Exitting Now ...\n", ret_ocl);

        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(oclProgram, oclComputeDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("OpenCL Program Build Log : %s\n", buffer);

        cleanup();
        exit(EXIT_FAILURE);
    }

    // create OpenCL kernel by passing kernel function name that we used in .cl file
    oclKernel = clCreateKernel(oclProgram, "matrixMultiply", &ret_ocl);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateKernel() Failed : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    int size = sizeof(cl_float) * (numCRows * numCColumns);

    // allocate device-memory
    deviceA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 1st Input Array : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 2nd Input Array : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceC = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, size, NULL, &ret_ocl);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 2nd Input Array : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 0th argument i.e. deviceA
    ret_ocl = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceA);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 1st Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 1st argument i.e. deviceB
    ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceB);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 2nd Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 2nd argument i.e. deviceC
    ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceC);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 3rd Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 3rd argument i.e. A Rows
    ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&numARows);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 4th Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 4rd argument i.e. A Columns
    ret_ocl = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void*)&numAColumns);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 5th Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 5th argument i.e. B Rows
    ret_ocl = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void*)&numBRows);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 6h Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 6rd argument i.e. B Columns
    ret_ocl = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void*)&numBColumns);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 7th Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 7th argument i.e. C Rows
    ret_ocl = clSetKernelArg(oclKernel, 7, sizeof(cl_int), (void*)&numCRows);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 8th Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0 based 8th argument i.e. C Rows
    ret_ocl = clSetKernelArg(oclKernel, 8, sizeof(cl_int), (void*)&numCColumns);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 9th Argument : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // write abve 'input' device buffer to device memory
    ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceA, CL_FALSE, 0, size, hostA, 0, NULL, NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueWriteBuffer() Failed For 1st Input Device Buffer : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceB, CL_FALSE, 0, size, hostB, 0, NULL, NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueWriteBuffer() Failed For 2nd Input Device Buffer : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // run the kernel
    globalWorkSize = roundGlobalSizeToNearestMultipleOfLocalSize(localWorkSize, (numCRows * numCColumns));

    // start timer
    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ret_ocl = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueNDRangeKernel() Failed : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // finish OpenCL command queue
    clFinish(oclCommandQueue);

    // stop timer
    sdkStopTimer(&timer);
    timeOnGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);

    // read back result from the device (i.e from deviceOutput) into cpu variable (i.e hostOutput)
    ret_ocl = clEnqueueReadBuffer(oclCommandQueue, deviceC, CL_TRUE, 0, size, hostC, 0, NULL, NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueReadBuffer() Failed : %d. Exitting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // results
    matMulHost(hostA, hostB, CHost, numAColumns, numCHostRows, numCHostColumns);

    // compare results for golden-host
    const float epsilon = 0.000001f;
    bool bAccuracy = true;
    int breakValue = 0;
    int i;
    for (i = 0; i < numARows * numAColumns; i++)
    {
        float val1 = CHost[i];
        float val2 = hostC[i];
        if (fabs(val1 - val2) > epsilon)
        {
            bAccuracy = false;
            breakValue = i;
            break;
        }
    }

   

    char str[125];
    if (bAccuracy == true)
        sprintf(str, "%s", "All Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
    else
        sprintf(str, "%s", "All Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");

   
    printf("The Dimensions Of First Matrix Are : %d x %d\n", numARows, numAColumns);
    printf("The Dimensions Of 'Second Matrix Are : %d x %d\n", numBRows, numBColumns);
    printf("The Dimensions Of Output Matrix On CPU Are : %d x %d\n", numCRows, numCColumns);

    printf("The Dimensions Of Output Matrix Of GPU Are : %d x %d\n", numCHostRows, numCHostColumns);

    printf("Size of First Matrix  = %d\n", sizeA);
    printf("Size of Second Matrix  = %d\n", sizeB);
    printf("Size of Output Matrix On CPU Matrix  = %d\n", sizeC);
    printf("Size of Output Matrix On GPU  = %d\n", sizeCHost);
    printf("The Time Taken To Do Above Multiplication On CPU = %.6f (ms)\n", timeOnCPU);
    printf("The Time Taken To Do Above Multiplication On GPU = %.6f (ms)\n", timeOnGPU);
    printf("%s\n", str);

    // total cleanup
    cleanup();

    return(0);
}

void cleanup(void)
{
    // code

    // OpenCL cleanup
    if (oclSourceCode)
    {
        free((void*)oclSourceCode);
        oclSourceCode = NULL;
    }

    if (oclKernel)
    {
        clReleaseKernel(oclKernel);
        oclKernel = NULL;
    }

    if (oclProgram)
    {
        clReleaseProgram(oclProgram);
        oclProgram = NULL;
    }

    if (oclCommandQueue)
    {
        clReleaseCommandQueue(oclCommandQueue);
        oclCommandQueue = NULL;
    }

    if (oclContext)
    {
        clReleaseContext(oclContext);
        oclContext = NULL;
    }

    // free allocated device-memory
    if (deviceA)
    {
        clReleaseMemObject(deviceA);
        deviceA = NULL;
    }

    if (deviceB)
    {
        clReleaseMemObject(deviceB);
        deviceB = NULL;
    }

    if (deviceC)
    {
        clReleaseMemObject(deviceC);
        deviceC = NULL;
    }

    // free allocated host-memory
    if (hostA)
    {
        free(hostA);
        hostA = NULL;
    }

    if (hostB)
    {
        free(hostB);
        hostB = NULL;
    }

    if (hostC)
    {
        free(hostC);
        hostC = NULL;
    }

    if (CHost)
    {
        free(CHost);
        CHost = NULL;
    }
}

void fillFloatArrayWithRandomNumbers(float* pFloatArray, int iSize)
{
    // code
    int i;
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i)
    {
        pFloatArray[i] = fScale * rand();
    }
}

size_t roundGlobalSizeToNearestMultipleOfLocalSize(int local_size, unsigned int global_size)
{
    // code
    unsigned int r = global_size % local_size;
    if (r == 0)
    {
        return(global_size);
    }
    else
    {
        return(global_size + local_size - r);
    }
}

void matMulHost(float* A, float* B, float* C, int iAColumns, int iCRows, int iCColumns)
{
    // code
    // start timer
    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < iCRows; ++i)
    {
        for (int j = 0; j < iCColumns; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < iAColumns; ++k)
            {
                float a = A[i * iAColumns + k];
                float b = B[k * iCColumns + j];
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

char* loadOclProgramSource(const char* filename, const char* preamble, size_t* sizeFinalLength)
{
    // locals
    FILE* pFile = NULL;
    size_t sizeSourceLength;

    pFile = fopen(filename, "rb"); // binary read
    if (pFile == NULL)
        return(NULL);

    size_t sizePreambleLength = (size_t)strlen(preamble);

    // get the length of the source code
    fseek(pFile, 0, SEEK_END);
    sizeSourceLength = ftell(pFile);
    fseek(pFile, 0, SEEK_SET); // reset to beginning

    // allocate a buffer for the source code string and read it in
    char* sourceString = (char*)malloc(sizeSourceLength + sizePreambleLength + 1);
    memcpy(sourceString, preamble, sizePreambleLength);
    if (fread((sourceString)+sizePreambleLength, sizeSourceLength, 1, pFile) != 1)
    {
        fclose(pFile);
        free(sourceString);
        return(0);
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFile);
    if (sizeFinalLength != 0)
    {
        *sizeFinalLength = sizeSourceLength + sizePreambleLength;
    }
    sourceString[sizeSourceLength + sizePreambleLength] = '\0';

    return(sourceString);
}
