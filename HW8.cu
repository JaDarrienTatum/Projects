// nvcc HW8.cu -o temp

#include <sys/time.h>
#include <stdio.h>
#include "./MyCuda.h"

//Length of vectors to be added.
#define N 655350

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
	BlockSize.x = 1000;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(BlockSize.x*65535 < N)
	{
		printf("\n Your vector of length %d is too long to work in this code.\n Good Bye\n", BlockSize.x*65535);
		exit(0);
	}
	GridSize.x = (N - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,(N+11)*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);

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
		A_CPU[i] = (float)1;	
		B_CPU[i] = 1.0;
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); free(B_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
	myCudaErrorCheck(__FILE__, __LINE__);
}

//This is the kernel. It is the function that will run on the GPU.
__global__ void DotProductGPU(float *a, float *b, float *c, int n)
{
	int threadNumber = threadIdx.x;
	int vectorNumber = threadIdx.x + blockDim.x*blockIdx.x;
	//int temp;
	
	if(vectorNumber < n)
	{
		c[vectorNumber] = a[vectorNumber] * b[vectorNumber];
	}
	__syncthreads();
	int fold = blockDim.x;
	
	while (fold > 2 && threadNumber<fold/2)
	{
	
		if (fold %2 == 1)
		{
			if (threadNumber == 0)
			{
				c[vectorNumber] += c[vectorNumber + fold-1];
			}
			fold--;
		}
		__syncthreads();
		fold/= 2;
		if (threadNumber < fold)
		{
			c[vectorNumber] += c[vectorNumber + fold];
		}
		__syncthreads();
	}
	
}

int main()
{
	float dotProduct, temp;
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
	dotProduct = 0.0;
	printf("\n grid size = %d\n", GridSize.x);
	for(int k = 0; k < GridSize.x; k++)
	{
		cudaMemcpyAsync(&temp, &C_GPU[k * BlockSize.x], sizeof(float), cudaMemcpyDeviceToHost);
		myCudaErrorCheck(__FILE__, __LINE__);
		dotProduct += temp;
	}

	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	
	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time/1000.0));	

	//Displaying the dot product.
	printf("Dot product = %.15f\n", dotProduct);
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	return(0);
}
