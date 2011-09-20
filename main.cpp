#include <time.h>
#include <sys/time.h>
#include <stdio.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

#include "wave.h"

#define WIDTH 700
#define HEIGHT 700

#define WAVES_COUNT 4

Wave waves[WAVES_COUNT] =
{
    {320, 100, 3, 1, 0, 0},
    {320, 60, 5, 5, 2, 0},
    {160, 40, 2, 8, 1, 0},
    {80, 40, 1, -4, 1, 0}
};

unsigned char bitmap[WIDTH][HEIGHT];

void computeBitmap(float t)
{
    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = 0; j < HEIGHT; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < WAVES_COUNT; k++)
            {
                sum += computeWave(&waves[k], i, j, t);
            }
            bitmap[i][j] = sum / WAVES_COUNT  + (255 / 2);
        }
    }
}

void display(void)
{
    static int t = 0;
    timeval start, end_frame, end_compute;

    gettimeofday(&start, NULL);

    t += 10;
    computeBitmap((float) t);

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

    glutMainLoop();
}
