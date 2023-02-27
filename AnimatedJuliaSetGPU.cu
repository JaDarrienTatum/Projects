//nvcc AnimatedJuliaSetGPU.cu -o temp -lglut -lGL -lm

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "./MyCuda.h"

#define MAXMAG 10.0
#define MAXITERATIONS 2000

static int Window;
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

dim3 BlockSize, GridSize;
float *PixelsCPU, *PixelsGPU; 
float StepSizeX, StepSizeY;

float RealSeed = -0.824;
float ImaginarySeed = -0.1711;
float DeltaSeed = 0.01;

// prototyping functions
void Display();
void idle();
void KeyPressed(unsigned char , int , int );
__device__ float getEscapeValue(float , float , float , float );
__global__ void colorPixels(float * , float , float , float , float , float , float );
void makeFractal();
void adjustSeed();
void paintScreen();
void setup();

void display()
{
	paintScreen();
}

void idle()
{
	makeFractal();
	paintScreen();
	adjustSeed();
}

void KeyPressed(unsigned char key, int x, int y)
{	
	if(key == 'q')
	{
		glutDestroyWindow(Window);
		printf("\nw Good Bye\n");
		exit(0);
	}
	
	if(key == 'S')
	{
		DeltaSeed += 0.001;
		printf("DeltaSeed = %f\n", DeltaSeed);
	}
	if(key == 's')
	{
		DeltaSeed -= 0.001;
		printf("DeltaSeed = %f\n", DeltaSeed);
	}
	if(key == 'r') //reset
	{
		RealSeed = -0.824;
		ImaginarySeed = -0.1711;
		DeltaSeed = 0.01;
	}
}

__device__ float getEscapeValue(float x, float y, float a, float b)
{
	float tempX;
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	int count = 0;
	float mag = sqrt(x*x + y*y);
	while (mag < maxMag && count < maxCount) 
	{
		//Zn = Zo*Zo + C
		//or xn + yni = (xo + yoi)*(xo + yoi) + A + Bi
		//xn = xo*xo - yo*yo + A (real Part)
		//yn = 2*xo*yo + B (imagenary part)
		
		//We will be changing the x but we need its old value to find y.	
		tempX = x; 
		x = x*x - y*y + a;
		y = sin((2.0 * tempX * y) + b);
		mag = exp(sqrt(x*x + y*y));
		count++;
	}
	return((float)count/(float)maxCount);
}

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, float a, float b) 
{
	float x, y;
		
	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	int id = 3*(threadIdx.x + blockDim.x*blockIdx.x);
	
	//Asigning each thread its x and y value of its pixel.
	x = xMin + dx*threadIdx.x;
	y = yMin + dy*blockIdx.x;
	
	float escapeValue = getEscapeValue(x, y, a, b);
	
	pixels[id] = escapeValue; //Setting the red
	if(0.99999 < escapeValue)
	{
		pixels[id]   = 0.0; //Setting the red
		pixels[id+1] = 0.0; //Setting the green
		pixels[id+2] = 1.0; //Setting the blue
	}
	else
	{
		pixels[id]   = 100*escapeValue; //Setting the red
		pixels[id+1] = 100*escapeValue; //Setting the green
		pixels[id+2] = 100*escapeValue; //Setting the blue
	}
}

void makeFractal()
{
	colorPixels<<<GridSize, BlockSize>>>(PixelsGPU, XMin, YMin, StepSizeX, StepSizeY, RealSeed, ImaginarySeed);
	myCudaErrorCheck(__FILE__, __LINE__);

	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(PixelsCPU, PixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	myCudaErrorCheck(__FILE__, __LINE__);
}

void adjustSeed()
{
	float temp;
	
	temp = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0; // Get random number between -1 at 1.
	RealSeed += DeltaSeed*temp;
	temp = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0; // Get random number between -1 at 1.
	ImaginarySeed += DeltaSeed*temp;
}

void paintScreen()
{
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, PixelsCPU); 
	glFlush();
}

void setup()
{
	//We need the 3 because each pixel has a red, green, and blue value.
	PixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&PixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	
	StepSizeX = (XMax - XMin)/((float)WindowWidth);
	StepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	//Threads in a block
	if(WindowWidth > 1024)
	{
	 	printf("The window width is too large to run with this program\n");
	 	printf("The window width width must be less than 1024.\n");
	 	printf("Good Bye and have a nice day!\n");
	 	exit(0);
	}
	BlockSize.x = WindowWidth;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//Blocks in a grid
	GridSize.x = WindowHeight;
	GridSize.y = 1;
	GridSize.z = 1;
	
	// Seading the random number generater.
	time_t t;
	srand((unsigned) time(&t));
}

int main(int argc, char** argv)
{ 
	setup();
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	Window = glutCreateWindow("Fractals man, fractals");
	glutKeyboardFunc(KeyPressed);
   	glutDisplayFunc(display);
	//glutReshapeFunc(reshape);
	//glutMouseFunc(mymouse);
	glutIdleFunc(idle);
   	glutMainLoop();
}
