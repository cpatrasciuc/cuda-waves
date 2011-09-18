#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

#define WIDTH 400
#define HEIGHT 400

unsigned char bitmap[WIDTH][HEIGHT];

void computeBitmap()
{
    for (int i = 0; i < WIDTH; i++)
    {
        if (i % 20 < 10)
        {
            for (int j = 0; j < HEIGHT; j++)
            {
                bitmap[i][j] = 255;
            }
        }
    }
}

void display(void)
{
    computeBitmap();
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f (1.0, 1.0, 1.0);
    glRasterPos2i(-1, -1);
    glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, bitmap);
    glFlush();
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
    glutDisplayFunc(display);
    
    glutMainLoop();
}
