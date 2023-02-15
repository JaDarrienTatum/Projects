// Dot product on one block
// nvcc HW7.cu -o temp

#include <sys/time.h>
#include <stdio.h>
#include "./MyCuda.h"

//Length of vectors to be added.
#define N 1000

//Function prototypes
void SetUpCudaDevices();
void AllocateMemory();
void Innitialize();
void CleanUp();
__global__ void DotProductGPU(float *, float *, float *, int );


//Globals
float *A_CPU, *B_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	BlockSize.x = N;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = 1.0;
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); free(B_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
}

//This is the kernel. It is the function that will run on the GPU.
//It adds vectors A and B then stores result in vector C
__global__ void DotProductGPU(float *a, float *b, float *c, int n)
{
	int id = threadIdx.x;
	*************************************************
	???
	************************************************
}

int main()
{
	float ???;
	timeval start, end;
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	//Starting the timer
	gettimeofday(&start, NULL);

	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	//Calling the Kernel (GPU) function.	
	DotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(???, &C_GPU[0], sizeof(float), cudaMemcpyDeviceToHost);
	myCudaErrorCheck(__FILE__, __LINE__);

	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	
	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time/1000.0));	

	//Displaying the dot product.
	printf("Dot product = %.15f\n", dot);
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	return(0);
}
