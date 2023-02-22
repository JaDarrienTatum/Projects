// nvcc HW9.cu -o temp

#include <sys/time.h>
#include <stdio.h>
#include "./MyCuda.h"

//Length of vectors to be added. Max int value is 2147483647
#define N 214748
#define BLOCK_SIZE 64


//Function prototypes
void SetUpCudaDevices();
void AllocateMemory();
void Innitialize();
void CleanUp();
__global__ void DotProductGPU(float *, float *, float *, int );
float dotProductCPU(float *, float *);

//Globals
float *A_CPU, *B_CPU; //CPU pointers
float *A_GPU, *B_GPU, *DotGPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices(int vectorSize)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	//long int maxThreads = prop.maxGridSize[0]*BLOCK_SIZE;
	
	//printf("\n prop.maxGridSize[0] = %d", prop.maxGridSize[0]);
	
	BlockSize.x = BLOCK_SIZE;
	if(prop.maxThreadsDim[0] < BlockSize.x)
	{
		printf("\n You are trying to create more threads (%d) than your GPU can suppport on a block (%d).\n Good Bye\n", BlockSize.x, prop.maxThreadsDim[0]);
		exit(0);
	}
	int temp = BlockSize.x;
	while(1 < temp)
	{
		if(temp%2 != 0)
		{
			printf("\n Your block size %d is not a power of 2 hence, this code will not work.\n Good Bye\n", BlockSize.x);
			exit(0);
		}
		temp /= 2;
	}
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	if(prop.maxGridSize[0] < GridSize.x)
	{
		printf("\n You are trying to create more blocks (%d) than your GPU can suppport (%d).\n Good Bye\n", GridSize.x, prop.maxGridSize[0]);
		exit(0);
	}
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory(float vectorSize)
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,vectorSize*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,vectorSize*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&DotGPU,sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(vectorSize*sizeof(float));
	B_CPU = (float*)malloc(vectorSize*sizeof(float));
	
	//Setting the vector to zero so the ectra values will not affect the dot product.
	memset(A_CPU, 0.0, vectorSize*sizeof(float));
	memset(B_CPU, 0.0, vectorSize*sizeof(float));
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)1;	
		B_CPU[i] = 1.0;
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); free(B_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(DotGPU);
	myCudaErrorCheck(__FILE__, __LINE__);
}

//This is the kernel. It is the function that will run on the GPU.
__global__ void DotProductGPU(float *a, float *b, float *DotGPU, int n)
{
	int threadNumber = threadIdx.x;
	int vectorNumber = threadIdx.x + blockDim.x*blockIdx.x;
	int fold = blockDim.x; 
	__shared__ float c_sh[BLOCK_SIZE];
	
	//***********************************
	
	if (vectorNumber < n)
		{
			c_sh[threadNumber] = a[vectorNumber] * b[vectorNumber];
		}
	__syncthreads();
		
	while (fold>=2 && threadNumber<fold/2)
		{
			fold = fold/2;
			if(threadNumber<fold && (vectorNumber+fold)<n)
			{
				c_sh[threadNumber] += c_sh[threadNumber + fold];
			}
		__syncthreads();
		}
	//***********************************
	
	if(threadNumber == 0) 
	{
		atomicAdd(DotGPU, c_sh[threadNumber]);
	}
}

float dotProductCPU(float *a, float *b)
{	
	int id;
	double sum = 0.0;
	
	for(id = 0; id < N; id++) sum += (a[id] * b[id]);
	return(sum);
}

int main()
{
	float dotCPU, dotGPU, time;
	timeval start, end;
	
	long int test = N;
	if(2147483647 < test)
	{
		printf("\nThe length of your vector is longer than the largest integer value allowed of 2147483647.\n");
		printf("You should check your code.\n Good Bye\n");
		exit(0);
	}
	
	int vectorSize = N + N%BLOCK_SIZE;
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices(vectorSize);

	//Partitioning off the memory that you will be using and padding with zero vector will be a factor of block size.
	AllocateMemory(vectorSize);

	//Loading up values to be added.
	Innitialize();
	
	gettimeofday(&start, NULL);
	dotCPU = dotProductCPU(A_CPU, B_CPU);
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\nTime for CPU dot product = %.15f milliseconds\n", (time/1000.0));
	printf("Dot product on CPU= %.15f\n", dotCPU);
	
	gettimeofday(&start, NULL);
	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	//Calling the Kernel (GPU) function.	
	DotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, DotGPU, vectorSize);
	myCudaErrorCheck(__FILE__, __LINE__);
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(&dotGPU, DotGPU, sizeof(float), cudaMemcpyDeviceToHost);
	myCudaErrorCheck(__FILE__, __LINE__);
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\nTime for GPU dot product = %.15f milliseconds\n", (time/1000.0));
	printf("Dot product on GPU = %.15f\n", dotGPU);
	
	printf("\nError = %.15f\n", dotCPU - dotGPU);
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	return(0);
}
