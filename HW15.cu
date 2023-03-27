// nvcc HW15.cu -o temp

#include <sys/time.h>
#include <stdio.h>
#include "./MyCuda.h"

#define DATA_CHUNKS (1024*1024) 
#define ENTIRE_DATA_SET (20*DATA_CHUNKS)
#define MAX_RANDOM_NUMBER 1000
#define BLOCK_SIZE 256

//Function prototypes
void setUpCudaDevices();
void allocateMemory();
void loadData();
void cleanUp();
__global__ void trigAdditionGPU(float *, float *, float *, int );

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A0_GPU, *B0_GPU, *C0_GPU, *A1_GPU, *B1_GPU, *C1_GPU; //GPU pointers
cudaEvent_t StartEvent, StopEvent;
cudaStream_t Stream0, Stream1;

//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	cudaDeviceProp prop;
	int whichDevice;
	
	cudaGetDevice(&whichDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	cudaGetDeviceProperties(&prop, whichDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	if(prop.deviceOverlap != 1)
	{
		printf("\n GPU will not handle overlaps so no speedup from streams");
		printf("\n Good bye.");
		exit(0);
	}
	
	cudaStreamCreate(&Stream0);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaStreamCreate(&Stream1);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0)
	{
		printf("\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye.");
		exit(0);
	}
	GridSize.x = DATA_CHUNKS/BLOCK_SIZE;
	GridSize.y = 1;
	GridSize.z = 1;	
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A0_GPU,DATA_CHUNKS*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B0_GPU,DATA_CHUNKS*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C0_GPU,DATA_CHUNKS*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&A1_GPU,DATA_CHUNKS*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B1_GPU,DATA_CHUNKS*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C1_GPU,DATA_CHUNKS*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	
	//Allocate page locked Host (CPU) Memory
	cudaHostAlloc(&A_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&B_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&C_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	myCudaErrorCheck(__FILE__, __LINE__);
}

void loadData()
{
	time_t t;
	srand((unsigned) time(&t));
	
	for(int i = 0; i < ENTIRE_DATA_SET; i++)
	{		
		A_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;
		B_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(A0_GPU); 
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B0_GPU); 
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C0_GPU); 
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaFree(A1_GPU); 
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B1_GPU); 
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C1_GPU); 
	myCudaErrorCheck(__FILE__, __LINE__);
	
	cudaFreeHost(A_CPU);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(B_CPU);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(C_CPU);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	cudaEventDestroy(StartEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	cudaStreamDestroy(Stream0);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaStreamDestroy(Stream1);
	myCudaErrorCheck(__FILE__, __LINE__);
}

__global__ void trigAdditionGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n)
	{
		c[id] = sin(a[id]) + cos(b[id]);
	}
}

int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	loadData();
	
	cudaEventRecord(StartEvent, 0);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS*2)
	{
		//******************************************
		cudaMemcpyAsync(A0_GPU, A_CPU + i, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice,Stream0);
		myCudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B0_GPU, B_CPU + i, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice,Stream0);
		myCudaErrorCheck(__FILE__, __LINE__);
		trigAdditionGPU<<<DATA_CHUNKS/BLOCK_SIZE,BLOCK_SIZE,0,Stream0>>>(A0_GPU, B0_GPU, C0_GPU, DATA_CHUNKS);
		cudaMemcpyAsync(C_CPU + i, C0_GPU,DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream0);
		myCudaErrorCheck(__FILE__, __LINE__);
		
		cudaMemcpyAsync(A1_GPU, A_CPU + i + DATA_CHUNKS, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice,Stream1);
		myCudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B1_GPU, B_CPU + i + DATA_CHUNKS, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice,Stream1);
		myCudaErrorCheck(__FILE__, __LINE__);
		trigAdditionGPU<<<DATA_CHUNKS/BLOCK_SIZE,BLOCK_SIZE,0,Stream1>>>(A1_GPU, B1_GPU, C1_GPU, DATA_CHUNKS);
		cudaMemcpyAsync(C_CPU + i + DATA_CHUNKS, C1_GPU,DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream1);
		myCudaErrorCheck(__FILE__, __LINE__);
		//******************************************
	}
	
	// Make the CPU wait until the Streams have finishd before it continues.
	cudaStreamSynchronize(Stream0);
	cudaStreamSynchronize(Stream1);
	
	cudaEventRecord(StopEvent, 0);
	myCudaErrorCheck(__FILE__, __LINE__);
	// Make the CPU wiat until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent); 
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU = %3.1f milliseconds", timeEvent);
	
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
