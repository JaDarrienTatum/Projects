// JaDarrien Tatum
// HW2 Vector addition on the CPU
//nvcc HW2_VectorAddition.cu -o VectorAddition
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Length of the vector
#define N 10 

//Global CPU pointers (floats)
//(??? I need 3 points to floats. You should name them so the rest of the code works.) 
float *A_CPU;
float *B_CPU;
float *C_CPU;

//Allocate Host (CPU) Memory
void AllocateMemory()
{					
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loads vectors that we will add.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)i;
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	cudaFree(A_CPU);
	cudaFree(B_CPU);
	cudaFree(C_CPU);
}

//Adds vectors A and B then stores result in vector C
void Addition(float *a, float *b, float *c, int n)
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
	c[i] = a[i] + b[i];	
		}
}

int main()
{
	int i;
	timeval start, end;
	
	//Partitioning off the memory that you will be using.	
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();

	//Starting the timer	
	gettimeofday(&start, NULL);

	//Add the two vectors
	Addition(A_CPU, B_CPU ,C_CPU, N);

	//Stopping the timer
	gettimeofday(&end, NULL);
	
	//Calculating the total time used in the addition and converting it to milliseconds
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);

	//Printing out the time to add the two vectors.
	printf("CPU Time in milliseconds= %.15f\n", (time/1000.0));
	
	// Displaying vector info you will want to comment out the vector print line when your
	// vector becomes big. This is just to make sure everything is running correctly before you do a big run.	
	for(i = 0; i < N; i++)		
	{		
		printf("A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", i, A_CPU[i], i, B_CPU[i], i, C_CPU[i]);
	}

	//Displaying the last value of the addition for a check when all vector display has been commented out.
	printf("Last Values are A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", N-1, A_CPU[N-1], N-1, B_CPU[N-1], N-1, C_CPU[N-1]);

	//Your done so cleanup your mess.	
	CleanUp();	
	
	return(0);
}
