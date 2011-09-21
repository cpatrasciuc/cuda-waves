#include <time.h>
#include <sys/time.h>
#include <stdio.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "wave.h"

#define WIDTH 640
#define HEIGHT 640

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
int *dev_t;
Wave *dev_waves;

dim3 grids(WIDTH/16, HEIGHT/16);
dim3 threads(16, 16);

__global__ void computeBitmap(int t, unsigned char *bitmap)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    float sum = 0.0;
    for (int k = 0; k < WAVES_COUNT; k++)
    {
        Wave *w = dev_waves[k];
        float tmp = std::sin((w->dx * x + w->dy * y) / w->wavelength +
                             t * w->speed * 0.01 + w->phase) + 1;
        sum += w->amplitude * tmp * tmp / 2.0;
    }
    bitmap[i][j] = sum / WAVES_COUNT  + (255 / 2);
}

void display(void)
{
    static int t = 0;
    timeval start, end_frame, end_compute;

    gettimeofday(&start, NULL);

    t += 10;
    cudaMemcpy(dev_t, &t, sizeof(int), cudaMemcpyHostToDevice);

    computeBitmap<<<grids, threads>>>(t, dev_bitmap);

    gettimeofday(&end_compute, NULL);

    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f (1.0, 1.0, 1.0);
    glRasterPos2i(-1, -1);
    glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, bitmap);
    glutSwapBuffers();
    glutPostRedisplay();

    gettimeofday(&end_frame, NULL);

    printf("Frame time: %d ms, Computation time: %d ms\n",
           (end_frame.tv_sec - start.tv_sec) * 1000 + (end_frame.tv_usec - start.tv_usec),
           (end_compute.tv_sec - start.tv_sec) * 1000 + (end_compute.tv_usec - start.tv_usec));
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("CUDA Demo");
    glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
    glClearColor (0.0, 1.0, 0.0, 0.0);
    glutIdleFunc(display);

    cudaMalloc((void**)&dev_bitmap, WIDTH * HEIGHT * sizeof(unsigned char));
    cudaMalloc((void**)&dev_t, sizeof(int));
    cudaMalloc((void**)&dev_waves, WAVES_COUNT * sizeof(Wave));

    cudaMemcpy(dev_waves, waves, WAVES_COUNT * sizeof(Wave));

    glutMainLoop();

    cudaFree(dev_bitmap);
    cudaFree(dev_t);
    cudaFree(dev_waves);
}
