#include <time.h>
#include <sys/time.h>
#include <stdio.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <GL/glext.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "wave.h"

#define WIDTH 1600
#define HEIGHT 1600

#define WAVES_COUNT 4

Wave waves[WAVES_COUNT] =
{
    {320, 100, 3, 1, 0, 0},
    {320, 60, 5, 5, 2, 0},
    {160, 40, 2, 8, 1, 0},
    {80, 40, 1, -4, 1, 0}
};

unsigned char bitmap[WIDTH][HEIGHT];

unsigned char *dev_bitmap;
int *dev_time;
__constant__ Wave dev_waves[WAVES_COUNT];

dim3 grids(WIDTH/16, HEIGHT/16);
dim3 threads(16, 16);

GLuint buffer;
cudaGraphicsResource *resource;

__global__ void computeBitmap(int *t, unsigned char *bitmap)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float sum = 0.0;
    for (int k = 0; k < WAVES_COUNT; k++)
    {
        Wave w = dev_waves[k];
        float tmp = sin((w.dx * x + w.dy * y) / w.wavelength +
                             *t * w.speed * 0.001 + w.phase) + 1;
        sum += w.amplitude * tmp * tmp / 2.0;
    }
    bitmap[x*WIDTH + y] = sum / WAVES_COUNT  + (255 / 2);
}

void display(void)
{
    static int t = 0;
    timeval start, end_frame, end_compute;
    cudaEvent_t start_event, stop_event;
    float cudaTime;

    gettimeofday(&start, NULL);
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    t += 10;
    cudaMemcpy(dev_time, &t, sizeof(int), cudaMemcpyHostToDevice);

    size_t size;
    cudaGraphicsMapResources(1, &resource);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_bitmap, &size, resource);

    computeBitmap<<<grids, threads>>>(dev_time, dev_bitmap);

    cudaGraphicsUnmapResources(1, &resource);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    gettimeofday(&end_compute, NULL);

    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f (1.0, 1.0, 1.0);
    glRasterPos2i(-1, -1);
    glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
    glutSwapBuffers();
    glutPostRedisplay();

    gettimeofday(&end_frame, NULL);

    cudaEventElapsedTime(&cudaTime, start_event, stop_event);

    printf("Frame time: %d ms, Computation time: %.1f ms\n",
           (end_frame.tv_sec - start.tv_sec) * 1000 + (end_frame.tv_usec - start.tv_usec),
           cudaTime);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

int main(int argc, char **argv)
{
    int device;
    cudaGetDevice(&device);
    cudaGLSetGLDevice(device);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("CUDA Demo");
    glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
    glClearColor (0.0, 1.0, 0.0, 0.0);
    glutIdleFunc(display);

    glGenBuffers( 1, &buffer );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, buffer );
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH * HEIGHT, NULL, GL_DYNAMIC_DRAW_ARB );

    cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsWriteDiscard);

    cudaMalloc((void**)&dev_time, sizeof(int));

    cudaMemcpyToSymbol(dev_waves, waves, WAVES_COUNT * sizeof(Wave));

    glutMainLoop();

    cudaGraphicsUnregisterResource(resource);
}
