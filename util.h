#ifndef UTIL_H
#define	UTIL_H

void printArray(int *a, int count, const char *name = "The array")
{
    printf("%s is: [", name);
    for (int i = 0; i < count - 1; i++)
    {
        printf("%d, ", a[i]);
    }
    printf("%d]\n", a[count - 1]);
}

#endif	/* UTIL_H */

