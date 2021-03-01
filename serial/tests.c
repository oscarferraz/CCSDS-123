

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "utils.h"

int main(int argc, char *argv[]){
    char *p;

    long long arg = 137438691328;
    long long op = 32;
    /*long long arg = strtol(argv[1], &p, 10);

    

    printf("arg=%lli\n", arg);
    float power2 = (float)(1/(float)(((long long)0x1) << arg));
    printf("rho positive=%.32f\n", power2);
    power2 = (float)(((long long)0x1) << (arg-(2*arg)));
    printf("rho negative=%.32f\n", power2);*/

    long long power2 = ((long long)0x1) << (op - 1);
    // I have to use the trick of shifting not of the op quantity altogether as
    // when op == 64 no shift is actually performed by the system
    printf("esa: %lli\n", ((arg + power2) - (((((arg + power2) >> (op - 1) >> 1)) << (op - 1)) << 1)) - power2);
   

 

    // I have to use the trick of shifting not of the op quantity altogether as
    // when op == 64 no shift is actually performed by the system
    arg-(long long)((power2 << 1)*((int)((arg+power2)/(power2 << 1))));
    printf("mine: %lli\n", arg-(long long)((power2 << 1)*((int)((arg+power2)/(power2 << 1)))));
    return(0);
}