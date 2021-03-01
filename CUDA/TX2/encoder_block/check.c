/*
Oscar Ferraz 7/2/2019

verifica se o ficheiro descomprimido é iual ao ficheiro inicial
*/

/*
USAGE
check <file1> <file2>
*/
//=======================================================================================================================================================================================================================00

#include <stdio.h>
#include <stdlib.h> 
#include <errno.h>
int main(int argc, char **argv){
    FILE *fp1, *fp2;
    unsigned short c, c1, c2, value;
    int i=0;
    char *p;
    int row, column, band; 
    //only 3 arguments allowed
    if(argc != 6) {
        printf("usage: ./check <file1> <file2> rows columns bands\n");
        return 0;
    }

    row = strtol(argv[3], &p, 10);
    column = strtol(argv[4], &p, 10);
    band = strtol(argv[5], &p, 10);
    
    fp1 = fopen (argv[1],"r");
    if (fp1 == NULL) {
        printf ("File1 not opended, errno = %d\n", errno);
        return 0;
    }

    fp2 = fopen (argv[2],"r");
    if (fp2 == NULL) {
        printf ("File2 not opended, errno = %d\n", errno);
        return 0;
    }

    for(int j=0;j<row*column*band;j++){
        c2 = fgetc(fp1);
        value = fgetc(fp2);
   
        if(value!=c2){
            printf("files not identical at %d\n",i);
        
            printf ("%hu\n",value);
            printf ("%hu\n",c2);
            printf("TEST FAILED\n");
            return 0;
        }
        if(i % 1000000 == 0)
            printf ("%d\n",i);
        i++;

    }
    printf("file read %d times\n",i);

    fclose(fp1);
    fclose(fp2);
    printf("TEST PASSED\n");
    return 0;
}