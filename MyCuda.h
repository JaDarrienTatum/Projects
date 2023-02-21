void myCudaErrorCheck(const char* file, int line)
{
	cudaError_t error;
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("\n CUDA message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line-1);
		exit(0);
	}
}
