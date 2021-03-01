/*
Oscar Ferraz 8/2/2019

*/
#ifndef PREDICTOR_CUDA_H
#define PREDICTOR_CUDA_H

#include <cuda_runtime.h> 

#include "utils.h"
#include "predictor.h"

/*#define thrds_x 2
#define thrds_y 512
#define thrds_z 1

#define NUM_STREAMS 56*/

typedef struct __align__(16) {
    unsigned short s0, s1, s2, s3, s4, s5, s6, s7;
}ushort8;

typedef struct __align__(16) {
    unsigned int s0, s1, s2, s3, s4, s5, s6, s7;
}uint8;

typedef struct __align__(16) {
    signed int s0, s1, s2, s3, s4, s5, s6, s7;
}int8;

typedef struct __align__(16) {
    signed char s0, s1, s2, s3, s4, s5, s6, s7;
}char8;


//======================================================================================================================================================================
// GPU compute local sum


    //**************************************************************************************************
    //streams ushort8
        __global__ void GPU_compute_prediction(ushort8* d_samples, ushort8 *d_mpr, unsigned short dim_x, unsigned short dim_y,unsigned short offset);
    
//======================================================================================================================================================================
extern void cuda_compute_prediction(input_feature_t input_params, predictor_config_t predictor_params, unsigned short int * samples);

//extern unsigned int *central_local_difference_cuda(input_feature_t input_params, predictor_config_t predictor_params,  unsigned short int * samples);

extern unsigned short int *samples_cuda_malloc(input_feature_t input_params);

extern ushort4 *ushort_to_ushort4(input_feature_t input_params, unsigned short int * samples );

extern ushort8 *ushort_to_ushort8(input_feature_t input_params, unsigned short int * samples );

extern unsigned int *uint4_to_uint(input_feature_t input_params, uint4 * samples );

extern signed int *int4_to_int(input_feature_t input_params, int4 * samples );

extern unsigned short *ushort4_to_ushort(input_feature_t input_params, ushort4* samples );

extern unsigned short *ushort8_to_ushort(input_feature_t input_params, ushort8* samples );

#endif