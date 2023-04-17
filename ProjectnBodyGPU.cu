//Optimized using shared memory and on chip memory	
//Initail conditions are setup in a cube.																																											
// nvcc ProjectnBodyGPU.cu -o nBodyGPU -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

#include <sys/time.h>
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 256

#define N 8000

#define XWindowSize 1000
#define YWindowSize 1000

#define DRAW 1000
#define DAMP 0.5

#define DT 0.001
#define STOP_TIME 2.0

#define G 1.0
#define H 1.0

#define EYE 10.0
#define FAR 50.0

// Globals
float4 Position[N], Velocity[N], Force[N];
float4 *PositionGPU, *VelocityGPU, *ForceGPU;
dim3 Block, Grid;

void set_initail_conditions()
{
	int i,j,k,num,particles_per_side;
	float position_start, temp;
	float initail_seperation;

	temp = pow((float)N,1.0/3.0) + 0.99999;
	particles_per_side = temp;
    	position_start = -(particles_per_side -1.0)/2.0;
	initail_seperation = 2.0;
	
	for(i=0; i<N; i++)
	{
		Position[i].w = 1.0;
	}
	
	num = 0;
	for(i=0; i<particles_per_side; i++)
	{
		for(j=0; j<particles_per_side; j++)
		{
			for(k=0; k<particles_per_side; k++)
			{
			    if(N <= num) break;
				Position[num].x = position_start + i*initail_seperation;
				Position[num].y = position_start + j*initail_seperation;
				Position[num].z = position_start + k*initail_seperation;
				Velocity[num].x = 0.0;
				Velocity[num].y = 0.0;
				Velocity[num].z = 0.0;
				num++;
			}
		}
	}
}

void setupDevice()
{
	Block.x = BLOCK_SIZE;
	Block.y = 1;
	Block.z = 1;
	
	Grid.x = (N-1)/Block.x + 1;
	Grid.y = 1;
	Grid.z = 1;
	
	cudaMalloc( (void**)&PositionGPU, N *sizeof(float4) );
	cudaMalloc( (void**)&VelocityGPU, N *sizeof(float4) );
	cudaMalloc( (void**)&ForceGPU, N *sizeof(float4) );
}

void draw_picture()
{
	int i;

	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(Position[i].x, Position[i].y, Position[i].z);
		glutSolidSphere(0.1,20,20);
		glPopMatrix();
	}

	glutSwapBuffers();
}
                                 
__device__ float4 getBodyBodyForce(float4 p0, float4 p1)
{
	float4 f;
	float dx = p1.x - p0.x;
	float dy = p1.y - p0.y;
	float dz = p1.z - p0.z;
	float r2 = dx*dx + dy*dy + dz*dz;
	float r = sqrt(r2);

	float force  = (G*p0.w*p1.w)/(r2) - (H*p0.w*p1.w)/(r2*r2);

	f.x = force*dx/r;
	f.y = force*dy/r;
	f.z = force*dz/r;

	return(f);
}

__global__ void getForces(float4 *pos, float4 *vel, float4 * force)
{
	int j,ii;
	float4 force_mag, forceSum;
	float4 posMe;
	__shared__ float4 shPos[BLOCK_SIZE];
	int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    	forceSum.x = 0.0;
	forceSum.y = 0.0;
	forceSum.z = 0.0;
		
	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	posMe.w = pos[id].w;
	    
	for(j=0; j < gridDim.x; j++)
	{
	shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
	__syncthreads();

	#pragma unroll 32
	for(int i=0; i < blockDim.x; i++)	
	{
		ii = i + blockDim.x*j;
		    if(ii != id && ii < N) 
		    {
		    	force_mag = getBodyBodyForce(posMe, shPos[i]);
			forceSum.x += force_mag.x;
			forceSum.y += force_mag.y;
			forceSum.z += force_mag.z;
		    }
	   	 }
	}
	if(id <N)
	{
	    force[id].x = forceSum.x;
	    force[id].y = forceSum.y;
	    force[id].z = forceSum.z;
	}
}

__global__ void moveBodies(float time, float4 *pos, float4 *vel, float4 * force)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id < N)
    {
		if(time == 0.0)
		{
			vel[id].x += ((force[id].x-DAMP*vel[id].x)/pos[id].w)*0.5*DT;
			vel[id].y += ((force[id].y-DAMP*vel[id].y)/pos[id].w)*0.5*DT;
			vel[id].z += ((force[id].z-DAMP*vel[id].z)/pos[id].w)*0.5*DT;
		}
		else
		{
			vel[id].x += ((force[id].x-DAMP*vel[id].x)/pos[id].w)*DT;
			vel[id].y += ((force[id].y-DAMP*vel[id].y)/pos[id].w)*DT;
			vel[id].z += ((force[id].z-DAMP*vel[id].z)/pos[id].w)*DT;
		}

		pos[id].x += vel[id].x*DT;
		pos[id].y += vel[id].y*DT;
		pos[id].z += vel[id].z*DT;
    }
}

void n_body()
{
	int   tdraw = 0; 
	float time = 0.0;
	
    	cudaMemcpy( PositionGPU, Position, N *sizeof(float4), cudaMemcpyHostToDevice );
    	cudaMemcpy( VelocityGPU, Velocity, N *sizeof(float4), cudaMemcpyHostToDevice );
	while(time < STOP_TIME)
	{	
		getForces<<<Grid, Block>>>(PositionGPU, VelocityGPU, ForceGPU);
		moveBodies<<<Grid, Block>>>(time, PositionGPU, VelocityGPU, ForceGPU);
        
		if(tdraw == DRAW) 
		{
			cudaMemcpy( Position, PositionGPU, N *sizeof(float4), cudaMemcpyDeviceToHost );
			draw_picture();
			printf("\n Time = %f \n", time);
			tdraw = 0;
		}
		time += DT;
		tdraw++;
	}
}

void control()
{	
	timeval start, end;
	double totalRunTime;
	
	set_initail_conditions();
	setupDevice();
	draw_picture();
	
	gettimeofday(&start, NULL);
    	n_body();
    	gettimeofday(&end, NULL);
    	
	totalRunTime = (end.tv_sec * 1000000.0 + end.tv_usec) - (start.tv_sec * 1000000.0 + start.tv_usec);
	printf("\n Totl run time = %5.15f seconds\n", (totalRunTime/1000000.0));
	
	printf("\n DONE \n");
	exit(0);
}

void Display(void)
{
	gluLookAt(EYE, EYE, EYE, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	control();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, FAR);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("n Body GPU");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
