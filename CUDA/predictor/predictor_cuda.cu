/*
Oscar Ferraz 8/2/2019

*/

    
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>      // CUDA Runtime Functions
#include <helper_cuda.h>
#include <unistd.h>
#include <sys/time.h>


struct timespec start, end;

extern "C" {
    #include "predictor_cuda.h"
}




//**************************************************************************************************
//streams ushort8
    __global__ void GPU_compute_prediction(ushort8 * d_samples, ushort8 *d_mpr, unsigned short dim_x, unsigned short dim_y, unsigned short offset){
        unsigned int x=threadIdx.x+blockIdx.x*blockDim.x;
        unsigned int y=threadIdx.y+blockIdx.y*blockDim.y;
        unsigned short z=offset+(threadIdx.z+blockIdx.z*blockDim.z);
        __shared__ ushort8 samples[thrds_x*thrds_y];
        ushort8 samplesx[1];
        uint8 local_sum[1];
        int8 scaled_predicted[1];
        int8 delta[1];
        ushort8 omega[1];
        uint8 abs_delta[1];
        char8 sign_scaled[1];
        __shared__ ushort8 mpr[thrds_x*thrds_y];

        

        //------------------------------------------------
        //leitura da imagem para shared memory
            reinterpret_cast<uint4*>(samples)[((threadIdx.x)*thrds_y)+threadIdx.y]=reinterpret_cast<uint4*>(d_samples)[(dim_x*dim_y*z)+(dim_x*y)+x];
            if(x>0 && y==0){
                reinterpret_cast<uint4*>(samplesx)[0]=reinterpret_cast<uint4*>(d_samples)[(dim_x*dim_y*z)+(dim_x*y)+x-1];
            }
            __syncthreads();

        //------------------------------------------------
        //local sum
            if(y>0){
                local_sum[0].s0=4*samples[((threadIdx.x)*thrds_y)+threadIdx.y-1].s0;
                local_sum[0].s1=4*samples[((threadIdx.x)*thrds_y)+threadIdx.y-1].s1;
                local_sum[0].s2=4*samples[((threadIdx.x)*thrds_y)+threadIdx.y-1].s2;
                local_sum[0].s3=4*samples[((threadIdx.x)*thrds_y)+threadIdx.y-1].s3;
                local_sum[0].s4=4*samples[((threadIdx.x)*thrds_y)+threadIdx.y-1].s4;
                local_sum[0].s5=4*samples[((threadIdx.x)*thrds_y)+threadIdx.y-1].s5;
                local_sum[0].s6=4*samples[((threadIdx.x)*thrds_y)+threadIdx.y-1].s6;
                local_sum[0].s7=4*samples[((threadIdx.x)*thrds_y)+threadIdx.y-1].s7;
            }
            else{
                local_sum[0].s0=4*samplesx[0].s0;
                local_sum[0].s1=4*samplesx[0].s1;
                local_sum[0].s2=4*samplesx[0].s2;
                local_sum[0].s3=4*samplesx[0].s3;
                local_sum[0].s4=4*samplesx[0].s4;
                local_sum[0].s5=4*samplesx[0].s5;
                local_sum[0].s6=4*samplesx[0].s6;
                local_sum[0].s7=4*samplesx[0].s7;
            }

        //------------------------------------------------
        //scaled predicted
            if(x>0 || y>0){
                scaled_predicted[0].s0= ((signed int)(16*(local_sum[0].s0-0x20000)) >>5)+0x10001;
                scaled_predicted[0].s1= ((signed int)(16*(local_sum[0].s1-0x20000)) >>5)+0x10001;
                scaled_predicted[0].s2= ((signed int)(16*(local_sum[0].s2-0x20000)) >>5)+0x10001;
                scaled_predicted[0].s3= ((signed int)(16*(local_sum[0].s3-0x20000)) >>5)+0x10001;
                scaled_predicted[0].s4= ((signed int)(16*(local_sum[0].s4-0x20000)) >>5)+0x10001;
                scaled_predicted[0].s5= ((signed int)(16*(local_sum[0].s5-0x20000)) >>5)+0x10001;
                scaled_predicted[0].s6= ((signed int)(16*(local_sum[0].s6-0x20000)) >>5)+0x10001;
                scaled_predicted[0].s7= ((signed int)(16*(local_sum[0].s7-0x20000)) >>5)+0x10001;

                if(scaled_predicted[0].s0 < 0)
                    scaled_predicted[0].s0 = 0;
                if(scaled_predicted[0].s1 < 0)
                    scaled_predicted[0].s1 = 0;
                if(scaled_predicted[0].s2  < 0)
                    scaled_predicted[0].s2  = 0;
                if(scaled_predicted[0].s3 < 0)
                    scaled_predicted[0].s3 = 0;
                if(scaled_predicted[0].s4 < 0)
                    scaled_predicted[0].s4 = 0;
                if(scaled_predicted[0].s5 < 0)
                    scaled_predicted[0].s5 = 0;
                if(scaled_predicted[0].s6  < 0)
                    scaled_predicted[0].s6  = 0;
                if(scaled_predicted[0].s7 < 0)
                    scaled_predicted[0].s7 = 0;
            
                if(scaled_predicted[0].s0 > 0x10000)
                    scaled_predicted[0].s0 = 0x10000;
                if(scaled_predicted[0].s1 > 0x10000)
                    scaled_predicted[0].s1 = 0x10000;
                if(scaled_predicted[0].s2 > 0x10000)
                    scaled_predicted[0].s2 = 0x10000;
                if(scaled_predicted[0].s3 > 0x10000)
                    scaled_predicted[0].s3 = 0x10000;
                if(scaled_predicted[0].s4 > 0x10000)
                    scaled_predicted[0].s4 = 0x10000;
                if(scaled_predicted[0].s5 > 0x10000)
                    scaled_predicted[0].s5 = 0x10000;
                if(scaled_predicted[0].s6 > 0x10000)
                    scaled_predicted[0].s6 = 0x10000;
                if(scaled_predicted[0].s7 > 0x10000)
                    scaled_predicted[0].s7 = 0x10000;
            }
            else{
                scaled_predicted[0].s0 = 0x10000;
                scaled_predicted[0].s1 = 0x10000;
                scaled_predicted[0].s2 = 0x10000;
                scaled_predicted[0].s3 = 0x10000;
                scaled_predicted[0].s4 = 0x10000;
                scaled_predicted[0].s5 = 0x10000;
                scaled_predicted[0].s6 = 0x10000;
                scaled_predicted[0].s7 = 0x10000;
            } 
            
        //------------------------------------------------
        //delta
            delta[0].s0=samples[((threadIdx.x)*thrds_y)+threadIdx.y].s0-(    scaled_predicted[0].s0>>1);
            delta[0].s1=samples[((threadIdx.x)*thrds_y)+threadIdx.y].s1-(scaled_predicted[0].s1>>1);
            delta[0].s2=samples[((threadIdx.x)*thrds_y)+threadIdx.y].s2-(scaled_predicted[0].s2>>1);
            delta[0].s3=samples[((threadIdx.x)*thrds_y)+threadIdx.y].s3-(scaled_predicted[0].s3>>1);
            delta[0].s4=samples[((threadIdx.x)*thrds_y)+threadIdx.y].s4-(scaled_predicted[0].s4>>1);
            delta[0].s5=samples[((threadIdx.x)*thrds_y)+threadIdx.y].s5-(scaled_predicted[0].s5>>1);
            delta[0].s6=samples[((threadIdx.x)*thrds_y)+threadIdx.y].s6-(scaled_predicted[0].s6>>1);
            delta[0].s7=samples[((threadIdx.x)*thrds_y)+threadIdx.y].s7-(scaled_predicted[0].s7>>1);

        //------------------------------------------------
        //omega
            omega[0].s0=(  scaled_predicted[0].s0>>1);
            omega[0].s1=(scaled_predicted[0].s1>>1);
            omega[0].s2=(scaled_predicted[0].s2>>1);
            omega[0].s3=(scaled_predicted[0].s3>>1);
            omega[0].s4=(scaled_predicted[0].s4>>1);
            omega[0].s5=(scaled_predicted[0].s5>>1);
            omega[0].s6=(scaled_predicted[0].s6>>1);
            omega[0].s7=(scaled_predicted[0].s7>>1);

            if(omega[0].s0 > 0xFFFF -   (scaled_predicted[0].s0>>1))
                omega[0].s0 = 0xFFFF -  (scaled_predicted[0].s0>>1);
            if(omega[0].s1 > 0xFFFF - (scaled_predicted[0].s1>>1))
                omega[0].s1 = 0xFFFF -(scaled_predicted[0].s1>>1);
            if(omega[0].s2 > 0xFFFF - (scaled_predicted[0].s2>>1))
                omega[0].s2 = 0xFFFF -(scaled_predicted[0].s2>>1);
            if(omega[0].s3 > 0xFFFF - (scaled_predicted[0].s3>>1))
                omega[0].s3 = 0xFFFF -(scaled_predicted[0].s3>>1);
            if(omega[0].s4 > 0xFFFF - (scaled_predicted[0].s4>>1))
                omega[0].s4 = 0xFFFF -(scaled_predicted[0].s4>>1);
            if(omega[0].s5 > 0xFFFF - (scaled_predicted[0].s5>>1))
                omega[0].s5 = 0xFFFF -(scaled_predicted[0].s5>>1);
            if(omega[0].s6 > 0xFFFF - (scaled_predicted[0].s6>>1))
                omega[0].s6 = 0xFFFF -(scaled_predicted[0].s6>>1);
            if(omega[0].s7 > 0xFFFF - (scaled_predicted[0].s7>>1))
                omega[0].s7 = 0xFFFF -(scaled_predicted[0].s7>>1);

        //------------------------------------------------
        //sign scaled
            sign_scaled[0].s0 = (scaled_predicted[0].s0 & 0x1) != 0 ? -1 : 1;
            sign_scaled[0].s1 = (scaled_predicted[0].s1 & 0x1) != 0 ? -1 : 1;
            sign_scaled[0].s2 = (scaled_predicted[0].s2 & 0x1) != 0 ? -1 : 1;
            sign_scaled[0].s3 = (scaled_predicted[0].s3 & 0x1) != 0 ? -1 : 1;
            sign_scaled[0].s4 = (scaled_predicted[0].s4 & 0x1) != 0 ? -1 : 1;
            sign_scaled[0].s5 = (scaled_predicted[0].s5 & 0x1) != 0 ? -1 : 1;
            sign_scaled[0].s6 = (scaled_predicted[0].s6 & 0x1) != 0 ? -1 : 1;
            sign_scaled[0].s7 = (scaled_predicted[0].s7 & 0x1) != 0 ? -1 : 1;

        //------------------------------------------------
        //abs delta
            abs_delta[0].s0 = delta[0].s0 < 0 ? (-1*delta[0].s0) : delta[0].s0;
            abs_delta[0].s1 = delta[0].s1 < 0 ? (-1*delta[0].s1) : delta[0].s1;
            abs_delta[0].s2 = delta[0].s2 < 0 ? (-1*delta[0].s2) : delta[0].s2;
            abs_delta[0].s3 = delta[0].s3 < 0 ? (-1*delta[0].s3) : delta[0].s3;
            abs_delta[0].s4 = delta[0].s4 < 0 ? (-1*delta[0].s4) : delta[0].s4;
            abs_delta[0].s5 = delta[0].s5 < 0 ? (-1*delta[0].s5) : delta[0].s5;
            abs_delta[0].s6 = delta[0].s6 < 0 ? (-1*delta[0].s6) : delta[0].s6;
            abs_delta[0].s7 = delta[0].s7 < 0 ? (-1*delta[0].s7) : delta[0].s7;

        

        //------------------------------------------------
        //mpr
            if(abs_delta[0].s0 > omega[0].s0){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s0 = abs_delta[0].s0 + omega[0].s0;
            }
            else if(((sign_scaled[0].s0*delta[0].s0) <= omega[0].s0) && ((sign_scaled[0].s0*delta[0].s0) >= 0)){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s0 = 2*abs_delta[0].s0;
            }
            else{
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s0 = 2*abs_delta[0].s0 - 1;
            }

            if(abs_delta[0].s1 > omega[0].s1){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s1 = abs_delta[0].s1 + omega[0].s1;
            }
            else if(((sign_scaled[0].s1*delta[0].s1) <= omega[0].s1) && ((sign_scaled[0].s1*delta[0].s1) >= 0)){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s1 = 2*abs_delta[0].s1;
            }
            else{
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s1 = 2*abs_delta[0].s1 - 1;
            }

            if(abs_delta[0].s2 > omega[0].s2){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s2 = abs_delta[0].s2 + omega[0].s2;
            }
            else if(((sign_scaled[0].s2*delta[0].s2) <= omega[0].s2) && ((sign_scaled[0].s2*delta[0].s2) >= 0)){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s2 = 2*abs_delta[0].s2;
            }
            else{
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s2 = 2*abs_delta[0].s2 - 1;
            }

            if(abs_delta[0].s3 > omega[0].s3){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s3 = abs_delta[0].s3 + omega[0].s3;
            }
            else if(((sign_scaled[0].s3*delta[0].s3) <= omega[0].s3) && ((sign_scaled[0].s3*delta[0].s3) >= 0)){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s3 = 2*abs_delta[0].s3;
            }
            else{
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s3 = 2*abs_delta[0].s3 - 1;
            }

            if(abs_delta[0].s4 > omega[0].s4){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s4 = abs_delta[0].s4 + omega[0].s4;
            }
            else if(((sign_scaled[0].s4*delta[0].s4) <= omega[0].s4) && ((sign_scaled[0].s4*delta[0].s4) >= 0)){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s4 = 2*abs_delta[0].s4;
            }
            else{
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s4 = 2*abs_delta[0].s4 - 1;
            }

            if(abs_delta[0].s5 > omega[0].s5){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s5 = abs_delta[0].s5 + omega[0].s5;
            }
            else if(((sign_scaled[0].s5*delta[0].s5) <= omega[0].s5) && ((sign_scaled[0].s5*delta[0].s5) >= 0)){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s5 = 2*abs_delta[0].s5;
            }
            else{
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s5 = 2*abs_delta[0].s5 - 1;
            }

            if(abs_delta[0].s6 > omega[0].s6){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s6 = abs_delta[0].s6 + omega[0].s6;
            }
            else if(((sign_scaled[0].s6*delta[0].s6) <= omega[0].s6) && ((sign_scaled[0].s6*delta[0].s6) >= 0)){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s6 = 2*abs_delta[0].s6;
            }
            else{
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s6 = 2*abs_delta[0].s6 - 1;
            }

            if(abs_delta[0].s7 > omega[0].s7){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s7 = abs_delta[0].s7 + omega[0].s7;
            }
            else if(((sign_scaled[0].s7*delta[0].s7) <= omega[0].s7) && ((sign_scaled[0].s7*delta[0].s7) >= 0)){
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s7 = 2*abs_delta[0].s7;
            }
            else{
                mpr[((threadIdx.x)*thrds_y)+threadIdx.y].s7 = 2*abs_delta[0].s7 - 1;
            }

            reinterpret_cast<uint4*>(d_mpr)[(dim_x*dim_y*z)+(dim_x*y)+x]=reinterpret_cast<uint4*>(mpr)[((threadIdx.x)*thrds_y)+threadIdx.y];                   
        }

    


//makes the local sum
// ONLY FOR THE COLUMN-ORIENTED LOCAL SUM
extern "C" void cuda_compute_prediction(input_feature_t input_params, predictor_config_t predictor_params, unsigned short int * cuda_samples){
    
    
    
    ushort8 *samples=ushort_to_ushort8(input_params, cuda_samples);
        

    clock_gettime(CLOCK_MONOTONIC, &start);

    cudaError_t err=cudaSuccess; 

    //======================================================================================================================================================================
    //stream dimensions ushort8
        int nStreams=(input_params.z_size/8)/NUM_STREAMS;
        int streamSize=(input_params.x_size*input_params.y_size*(input_params.z_size/8))/nStreams;

    //======================================================================================================================================================================
    //stream initialization
        //**************************************************************************************************
        //streams ushort8
            cudaStream_t stream[nStreams];
            for (int i = 0; i < nStreams; ++i){
                err=cudaStreamCreate(&stream[i]);
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to create streams(error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                }
            }

    //======================================================================================================================================================================
    //kernel dimensions
        //**************************************************************************************************
        //streams ushort8
            dim3 threadsPerBlock(thrds_x,thrds_y,thrds_z);
            dim3 numBlocks(input_params.x_size/thrds_x,input_params.y_size/thrds_y,((input_params.z_size/thrds_z)/8)/nStreams);
    
    
    //======================================================================================================================================================================
    //size of variables
        //**************************************************************************************************
        //streams ushort8
            size_t streamBytes_in=sizeof(ushort8)*streamSize;
            size_t streamBytes_out=sizeof(ushort8)*streamSize;
            size_t size_samples=(sizeof(ushort8)*input_params.x_size*input_params.y_size*(input_params.z_size/8));

        
  

    //======================================================================================================================================================================
    //variables declaration
        //**************************************************************************************************
        //vectorized access ushort8 mpr output 
            ushort8 *d_samples=NULL;
            ushort8 *d_mpr=NULL;
            ushort8 *h_mpr4=NULL;
            #if defined (VERIFICATION)  || defined (TEX_VERIFICATION) // variables for verifing the result
                h_mpr=NULL;
            #endif

    //======================================================================================================================================================================
    //allocate host memory
        //**************************************************************************************************
        //vectorized access output
            err=cudaHostAlloc((void **)&h_mpr4, size_samples, cudaHostAllocDefault );
            if(err!=cudaSuccess){
                fprintf(stderr, "Failed to allocate host mpr(error code %d)!\n", cudaGetLastError());
                exit(EXIT_FAILURE);
            }

    //======================================================================================================================================================================
    //allocate device memory
        //**************************************************************************************************
        //shared memory mpr output
            //------------------------------------------------
            //samples
                err=cudaMalloc((void **)&d_samples, size_samples);
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to allocate device samples (error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                }

            //------------------------------------------------
            //mpr
                err=cudaMalloc((void **)&d_mpr, size_samples);
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to allocate device mpr (error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                }


    //======================================================================================================================================================================
    //copy data to device  
        //**************************************************************************************************
        //streams
            for (int i = 0; i < nStreams; ++i) {
                int offset = i * streamSize;
                err=cudaMemcpyAsync(&d_samples[offset], &samples[offset], streamBytes_in, cudaMemcpyHostToDevice, stream[i]);
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to copy the image samples from host to device (error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                }
            }


    //======================================================================================================================================================================
    //execute the kernel
        //**************************************************************************************************
        //streams
            for (int i = 0; i < nStreams; ++i) {
                GPU_compute_prediction<<<numBlocks, threadsPerBlock,0, stream[i]>>>(d_samples, d_mpr, input_params.x_size,input_params.y_size,  i*NUM_STREAMS);
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to launch the kernel for the calculation of the local sum (error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                } 
            }

    //======================================================================================================================================================================
    //copy the data from device to host
        //**************************************************************************************************
        //streams
            for (int i = 0; i < nStreams; ++i) {
                int offset = i * streamSize;
                err=cudaMemcpyAsync(&h_mpr4[offset], &d_mpr[offset], streamBytes_out, cudaMemcpyDeviceToHost, stream[i]);
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to copy the scaled from device to host (error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                }
            }


    //======================================================================================================================================================================
    //free the device memory
        //**************************************************************************************************
        //shared memory sign scaled
            //------------------------------------------------
            //samples
                err=cudaFree(d_samples);
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to free the samples from the device (error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                }

            //------------------------------------------------
            //mpr
                err=cudaFree(d_mpr);
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to free the mpr from the device (error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                }
    //======================================================================================================================================================================
    //free the host memory
        //**************************************************************************************************
        //vectorized access output
            #if !defined (VERIFICATION)  && !defined (TEX_VERIFICATION) // variables for verifing the result
                err=cudaFreeHost(h_mpr4);
                printf("delete\n");
                if(err!=cudaSuccess){
                    fprintf(stderr, "Failed to free the omega from the host (error code %d)!\n", cudaGetLastError());
                    exit(EXIT_FAILURE);
                }
            #endif
    
    //======================================================================================================================================================================
    //Destroy cuda streams
        for (int i = 0; i < nStreams; ++i){
            err=cudaStreamDestroy(stream[i]);
            if(err!=cudaSuccess){
                fprintf(stderr, "Failed to destroy streams(error code %d)!\n", cudaGetLastError());
                exit(EXIT_FAILURE);
            }
        }

    //======================================================================================================================================================================
    //save total time
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("%lf\n",(end.tv_sec-start.tv_sec)*1e3+(end.tv_nsec-start.tv_nsec)*1e-6);
    FILE *fp;
    fp = fopen("tempos_total.txt","a");
    if(fp == NULL){
        printf("Error writng file!\n");   
        exit(1);             
    }
   fprintf(fp,"%lf\n",(end.tv_sec-start.tv_sec)*1e3+(end.tv_nsec-start.tv_nsec)*1e-6);
   fclose(fp); 

    //**************************************************************************************************
    //vectorized access output

        #if defined (VERIFICATION)  || defined (TEX_VERIFICATION) // variables for verifing the result
            h_mpr=ushort8_to_ushort(input_params, h_mpr4);
            printf("file is being written\n");
            err=cudaFreeHost(h_mpr4);
            if(err!=cudaSuccess){
                fprintf(stderr, "Failed to free the scaled predicted from the host (error code %d)!\n", cudaGetLastError());
                exit(EXIT_FAILURE);
            }

        #endif
    return ;
}



//converts from short to short4 by band
extern "C" ushort4 *ushort_to_ushort4(input_feature_t input_params, unsigned short int * samples ){
    cudaError_t err=cudaSuccess;
    struct cudaDeviceProp prop;
    
    ushort4 *cuda_samples=NULL;
    size_t size=sizeof( ushort4)*input_params.x_size*input_params.y_size*(input_params.z_size/4);

    //allow pinned memory

    cudaDeviceReset();
    cudaGetDeviceProperties(&prop, 0);
    if (prop.canMapHostMemory==0) 
        cudaSetDeviceFlags(cudaDeviceMapHost);


    /*int device;
    cudaGetDevice(&device);

    cudaGetDeviceProperties(&prop, device);
    printf("pror %d, size=%d\n", prop.canMapHostMemory  , prop.concurrentKernels);
    //aloccate the image on pinned memory
    cuda_samples=(ushort4 *)malloc(size);*/
    err=cudaHostAlloc((void **)&cuda_samples, size, cudaHostAllocDefault /*|*//* cudaHostAllocMapped /*|*/ /*cudaHostAllocPortable /*|*/ /*cudaHostAllocWriteCombined*/);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate ushort8 samples(error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    int z,y,x,a=0;
    for(z = 0; z < input_params.z_size; z=z+4){
        for(y = 0; y < input_params.y_size; y++){
            for(x = 0; x < input_params.x_size; x++){
                cuda_samples[a].x=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x];
                cuda_samples[a].y=samples[(input_params.x_size*input_params.y_size*(z+1))+(input_params.x_size*y)+x];
                cuda_samples[a].z=samples[(input_params.x_size*input_params.y_size*(z+2))+(input_params.x_size*y)+x];
                cuda_samples[a].w=samples[(input_params.x_size*input_params.y_size*(z+3))+(input_params.x_size*y)+x];
                a++;
            }
        }
    }

    return (cuda_samples);
}

//converts from short to short8 by band
extern "C" ushort8 *ushort_to_ushort8(input_feature_t input_params, unsigned short int * samples ){
    cudaError_t err=cudaSuccess;
    cudaDeviceProp prop;
    
    ushort8 *cuda_samples=NULL;
    size_t size=sizeof( ushort8)*input_params.x_size*input_params.y_size*(input_params.z_size/8);

    //allow pinned memory
    cudaGetDeviceProperties(&prop, 0);
    if (prop.canMapHostMemory==0) 
        cudaSetDeviceFlags(cudaDeviceMapHost);

    //aloccate the image on pinned memory
    err=cudaHostAlloc((void **)&cuda_samples,size, cudaHostAllocDefault);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate ushort8 samples(error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    int z,y,x,a=0;
    for(z = 0; z < input_params.z_size; z=z+8){
        for(y = 0; y < input_params.y_size; y++){
            for(x = 0; x < input_params.x_size; x++){
                cuda_samples[a].s0=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x];
                cuda_samples[a].s1=samples[(input_params.x_size*input_params.y_size*(z+1))+(input_params.x_size*y)+x];
                cuda_samples[a].s2=samples[(input_params.x_size*input_params.y_size*(z+2))+(input_params.x_size*y)+x];
                cuda_samples[a].s3=samples[(input_params.x_size*input_params.y_size*(z+3))+(input_params.x_size*y)+x];
                cuda_samples[a].s4=samples[(input_params.x_size*input_params.y_size*(z+4))+(input_params.x_size*y)+x];
                cuda_samples[a].s5=samples[(input_params.x_size*input_params.y_size*(z+5))+(input_params.x_size*y)+x];
                cuda_samples[a].s6=samples[(input_params.x_size*input_params.y_size*(z+6))+(input_params.x_size*y)+x];
                cuda_samples[a].s7=samples[(input_params.x_size*input_params.y_size*(z+7))+(input_params.x_size*y)+x];

                a++;
            }
        }
    }

    return (cuda_samples);    
}

//convert uint4 to uint by band
extern "C" unsigned int *uint4_to_uint(input_feature_t input_params, uint4* samples ){
    
    unsigned int *cuda_samples=NULL;
    size_t size=sizeof( unsigned int )*input_params.x_size*input_params.y_size*input_params.z_size;

    cuda_samples=(unsigned int *)malloc(size);
    int z,y,x,a=0;
    for(z = 0; z < input_params.z_size/4; z++){
        for(y = 0; y < input_params.y_size; y++){
            for(x = 0; x < input_params.x_size; x++){
                cuda_samples[(input_params.x_size*input_params.y_size*a)+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].x;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+1))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].y;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+2))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].z;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+3))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].w;                
            }
        }
        a=a+4;
    }
    return (cuda_samples);
}

//convert int4 to int by band
extern "C" signed int *int4_to_int(input_feature_t input_params, int4* samples ){
    
    signed int *cuda_samples=NULL;
    size_t size=sizeof( signed int )*input_params.x_size*input_params.y_size*input_params.z_size;

    cuda_samples=(signed int *)malloc(size);
    int z,y,x,a=0;
    for(z = 0; z < input_params.z_size/4; z++){
        for(y = 0; y < input_params.y_size; y++){
            for(x = 0; x < input_params.x_size; x++){
                cuda_samples[(input_params.x_size*input_params.y_size*a)+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].x;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+1))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].y;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+2))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].z;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+3))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].w;                
            }
        }
        a=a+4;
    }
    return (cuda_samples);
}

//convert ushort4 to ushort by band
extern "C" unsigned short *ushort4_to_ushort(input_feature_t input_params, ushort4* samples ){
    
    unsigned short *cuda_samples=NULL;
    size_t size=sizeof( unsigned short )*input_params.x_size*input_params.y_size*input_params.z_size;

    cuda_samples=(unsigned short *)malloc(size);
    int z,y,x,a=0;
    for(z = 0; z < input_params.z_size/4; z++){
        for(y = 0; y < input_params.y_size; y++){
            for(x = 0; x < input_params.x_size; x++){
                cuda_samples[(input_params.x_size*input_params.y_size*a)+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].x;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+1))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].y;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+2))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].z;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+3))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].w;                
            }
        }
        a=a+4;
    }
    return (cuda_samples);
}

//convert ushort8 to ushort by band
extern "C" unsigned short *ushort8_to_ushort(input_feature_t input_params, ushort8* samples ){
    
    unsigned short *cuda_samples=NULL;
    size_t size=sizeof( unsigned short )*input_params.x_size*input_params.y_size*input_params.z_size;

    cuda_samples=(unsigned short *)malloc(size);
    int z,y,x,a=0;
    for(z = 0; z < input_params.z_size/8; z++){
        for(y = 0; y < input_params.y_size; y++){
            for(x = 0; x < input_params.x_size; x++){
                cuda_samples[(input_params.x_size*input_params.y_size*a)+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].s0;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+1))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].s1;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+2))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].s2;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+3))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].s3; 
                cuda_samples[(input_params.x_size*input_params.y_size*(a+4))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].s4;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+5))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].s5;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+6))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].s6;
                cuda_samples[(input_params.x_size*input_params.y_size*(a+7))+(input_params.x_size*y)+x]=samples[(input_params.x_size*input_params.y_size*z)+(input_params.x_size*y)+x].s7;               
            }
        }
        a=a+8;
    }
    return (cuda_samples);
}