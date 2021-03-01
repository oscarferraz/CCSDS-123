/*
Luca Fossati (Luca.Fossati@esa.int), European Space Agency

Software distributed under the "European Space Agency Public License � v2.0".

All Distribution of the Software and/or Modifications, as Source Code or Object Code,
must be, as a whole, under the terms of the European Space Agency Public License � v2.0.
If You Distribute the Software and/or Modifications as Object Code, You must:
(a)	provide in addition a copy of the Source Code of the Software and/or
Modifications to each recipient; or
(b)	make the Source Code of the Software and/or Modifications freely accessible by reasonable
means for anyone who possesses the Object Code or received the Software and/or Modifications
from You, and inform recipients how to obtain a copy of the Source Code.

The Software is provided to You on an �as is� basis and without warranties of any
kind, including without limitation merchantability, fitness for a particular purpose,
absence of defects or errors, accuracy or non-infringement of intellectual property
rights.
Except as expressly set forth in the "European Space Agency Public License � v2.0",
neither Licensor nor any Contributor shall be liable, including, without limitation, for direct, indirect,
incidental, or consequential damages (including without limitation loss of profit),
however caused and on any theory of liability, arising in any way out of the use or
Distribution of the Software or the exercise of any rights under this License, even
if You have been advised of the possibility of such damages.

*****************************************************************************************
Entropy Encoder as described in Section 5 of the CCSDS 123.0-R-1,
White Book Draft Recommendation for Space Data System Standards on
LOSSLESS MULTISPECTRAL & HYPERSPECTRAL IMAGE COMPRESSION
as of 09/11/2011.
*/

#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#define NUM_THREADS 12

#define DENVER 327520
#define ARM 111312


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <semaphore.h>
#include <cuda_runtime.h>      // CUDA Runtime Functions
#include <helper_cuda.h>
#include <sys/time.h>



extern "C" {
    #include "predictor.h"
    #include "entropy_encoder.h"
    #include "utils.h"
    

}

    extern unsigned int *h_len;
    extern int *h_code;



#include<sched.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <assert.h>
#include <pthread.h>


struct timespec start1, end1, start2, end2, start3, end3, start4, end4, start5, end5;
  


pthread_t threads[NUM_THREADS];
int thread_args[NUM_THREADS];
pthread_attr_t threads_attr[NUM_THREADS];
int i;
int result_code;
cpu_set_t cpusetp, cpusetp2, cpuset_zero, cpuset_sec_ext, cpuset_k, cpuset_stream0, cpuset_stream1, cpuset_stream2, cpuset_stream3, cpuset_stream4, cpuset_stream5, cpuset_stream6;


struct thread_args_second *args_second=NULL ;
struct thread_args_split *args_ksplit=NULL ;
struct thread_args_zero *args_zero=NULL ;
struct thread_args_sec_ext *args_sec_ext=NULL ;
struct thread_args_k *args_k=NULL ;
struct thread_args_stream_conc *args_stream0=NULL ;
struct thread_args_stream *args_stream1=NULL ;
struct thread_args_stream *args_stream2=NULL ;
struct thread_args_stream *args_stream3=NULL ;
struct thread_args_stream *args_stream4=NULL ;
struct thread_args_stream *args_stream5=NULL ;
struct thread_args_stream *args_stream6=NULL ;

struct thread_args_stream_conc *args_sample0=NULL ;
struct thread_args_sample *args_sample1=NULL ;
struct thread_args_sample *args_sample2=NULL ;
struct thread_args_sample *args_sample3=NULL ;
struct thread_args_sample *args_sample4=NULL ;
struct thread_args_sample *args_sample5=NULL ;
struct thread_args_sample *args_sample6=NULL ;

sem_t sem_zero;
sem_t sem_sec_ext;
sem_t sem_k;

extern unsigned int const_zero;
extern unsigned int const_less;
extern unsigned int const_diff_zero;
extern unsigned int zero;
extern unsigned int less;
extern unsigned int diff_zero;
extern unsigned int for1;
extern unsigned int for2;
extern unsigned int big_zero;


 int * aux=NULL;

 //unsigned int bytes[1100288];
 //unsigned int bits[1100288];



/******************************************************
* Routines for the Sample Adaptive Encoder
*******************************************************/
/* __global__ void GPU_compute_k_split_enc(unsigned short *d_mpr, unsigned int * d_len, int * d_code, int offset){
    unsigned int x=(threadIdx.x*64)+offset*65536;
    //if(threadIdx.x==0 && offset==1)
        //printf("x=%u\n",x);

       //if(x<70418369){

       unsigned short mpr[64];
       unsigned int len[1];
       unsigned short code[1];

       //------------------------------------------------
       //leitura da imagem para shared memory
       for(int i=0; i<64; i++)
           mpr[i]=d_mpr[x+i];

       unsigned int code_len_temp = 0;
       
       len[0] = (unsigned int)-1;
       code[0]=0;
       for(int k = 0; k < 14; k++){
           code_len_temp = 0;
           for(int i=0; i < 64; i++){
               //if(threadIdx.x+blockIdx.x*64==4096 && k==0)
                   //printf("code_len_temp=%u i=%d, mpr=%d\n", threadIdx.x, blockIdx.x, x); */
               //printf(" i=%d, k=%d, value=%hu, shift=%hu\n",i,k,args->ptr_k[i], (args->ptr_k[i] >> k));
               /*code_len_temp += (mpr[i] >> k) + 1 + k;
           }
           //if(threadIdx.x+blockIdx.x*64==4096)
           //printf("code_len_temp=%u\n", code_len_temp);

           if(code_len_temp < len[0]){
               
               len[0] = code_len_temp;
               code[0] = k;
               
           }
       }

       d_len[threadIdx.x+offset*1024]=len[0];
       d_code[threadIdx.x+offset*1024]=code[0];

   //}

} */

    __device__ void gpu_bitStream_store(unsigned long * compressed_stream, unsigned int * written_bytes,
        unsigned int * written_bits, unsigned int num_bits_to_write, unsigned long bits_to_write, unsigned int offset, unsigned int z){
        //if(*written_bytes==67)
        //printf("written_bytes=%u, written_bits=%u, num_bits_to_write=%u, bits_to_write=%u \n", *written_bytes, *written_bits, num_bits_to_write, bits_to_write);
        bits_to_write=bits_to_write &((0x1 << num_bits_to_write)-1);
        if(bits_to_write==0){
            written_bytes[z]+=(written_bits[z] + num_bits_to_write)/64;
            written_bits[z]= (written_bits[z] +num_bits_to_write)%64;
            //zero++;
        }
        else{
            if(num_bits_to_write + (written_bits[z]) <64){
                compressed_stream[(written_bytes[z])+offset] |= (bits_to_write << (64-(written_bits[z])- num_bits_to_write));
                (written_bits[z])+=num_bits_to_write;
            //less++;
            }
            else{
                if(written_bits[z] !=0){
                    compressed_stream[(written_bytes[z])+offset] |= bits_to_write >> (num_bits_to_write -(64-(written_bits[z])));
                    num_bits_to_write-= 64 -(written_bits[z]);
                    written_bits[z] =0;
                    (written_bytes[z])++;
                    //diff_zero++;
                }
                
            
                if(num_bits_to_write >0){
                    bits_to_write=bits_to_write&((0x1 << num_bits_to_write)-1);
                    compressed_stream[(written_bytes[z])+offset] |= (bits_to_write << (64 -num_bits_to_write));
                    (written_bits[z])+=num_bits_to_write;
                    //big_zero++;
                }
            }
        }
    }

///Writes bitToRepeat a number of times equal to numBitsToWrite into compressedStream, starting at byte
///writtenBytes and in that byte at bit writtenBits. It also updates writtenBytes and
///writtenBits according to the number of bits written
///@param compressed_stream pointer to the array holding the stream cotnaining the compressed data
///@param written_bytes number of bytes so far written to the stream
///@param written_bits number of bits so far written to the stream
///@param num_bits_to_write number of times the bit in the least significant position of bit_to_repeat
///has to be added to the stream
///@param bit_to_repeat byte whose least significant bit is to be added to the stream num_bits_to_write times
    __device__ void gpu_bitStream_store_constant(unsigned long * compressed_stream, unsigned int * written_bytes,
        unsigned int * written_bits, unsigned int num_bits_to_write, unsigned char bit_to_repeat, unsigned int offset, unsigned int z){
        
        if(bit_to_repeat==0){
            written_bytes[z]+=(written_bits[z] + num_bits_to_write)/64;
            written_bits[z]= (written_bits[z] +num_bits_to_write)%64;
            //const_zero++;
        } 
        else{
            bit_to_repeat = 0x1;
            if(num_bits_to_write + (written_bits[z])<64){
            compressed_stream[(written_bytes[z])+offset] |= ((long)((0x2 << (num_bits_to_write -1))-1) << (64-(written_bits[z])-num_bits_to_write));
            written_bits[z]+= num_bits_to_write;
            //const_less++;
            }
            else{
                if(written_bits[z]!=0){
                    compressed_stream[(written_bytes[z])+offset] |= (0x2 << (64-(written_bits[z]))-1)-1;
                    num_bits_to_write -=64-(written_bits[z]);
                    written_bits[z]=0;
                    (written_bytes[z])++;
                    //const_diff_zero++;

                }
            }

        }

    }




/// Creates the header and adds it to the output stream.
extern "C" void create_header(unsigned int * written_bytes, unsigned int * written_bits, unsigned long * compressed_stream,
    input_feature_t input_params, predictor_config_t predictor_params, encoder_config_t encoder_params){
    /* IMAGE METADATA */
    // User defined data
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 8, 0);
    // x, y, z dimensions
    bitStream_store(compressed_stream, written_bytes, written_bits, 16, input_params.x_size);
    bitStream_store(compressed_stream, written_bytes, written_bits, 16, input_params.y_size);
    bitStream_store(compressed_stream, written_bytes, written_bits, 16, input_params.z_size);
    // Sample type
    if(input_params.signed_samples != 0)
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
    else
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
    // dynamic range
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, input_params.dyn_range);
    // Encoding Sample Order and interleaving
    if(encoder_params.out_interleaving == BSQ){
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 16, 0);
    }
    else{
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
        bitStream_store(compressed_stream, written_bytes, written_bits, 16, encoder_params.out_interleaving_depth);
    }
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
    // Out word size
    bitStream_store(compressed_stream, written_bytes, written_bits, 3, encoder_params.out_wordsize);
    // Encoder type
    if(encoder_params.encoding_method == SAMPLE)
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    else
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 10, 0);

    /* PREDICTOR METADATA */
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
    // prediction bands
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, predictor_params.user_input_pred_bands);
    // prediction mode
    if(predictor_params.full != 0)
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    else
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // local sum
    if(predictor_params.neighbour_sum != 0)
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    else
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // Register size
    bitStream_store(compressed_stream, written_bytes, written_bits, 6, predictor_params.register_size);
    // Weight resolution
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, predictor_params.weight_resolution - 4);
    // weight update scaling exponent change interval
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, ((unsigned int)log2((float)predictor_params.weight_interval)) - 4);
    // weight update scaling exponent initial parameter
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, predictor_params.weight_initial + 6);
    // weight update scaling exponent final parameter
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, predictor_params.weight_final + 6);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // weight initialization method and weight initialization table flag
    if(predictor_params.weight_init_table != NULL)
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 1);
    else
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
    // weight initialization resolution
    if(predictor_params.weight_init_table != NULL)
        bitStream_store(compressed_stream, written_bytes, written_bits, 5, predictor_params.weight_init_resolution);
    else
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 5, 0);
    // Weight initialization table
    if(predictor_params.weight_init_table != NULL){
        unsigned int z = 0, cz = 0;
        for(z = 0; z < input_params.z_size; z++){
            if(predictor_params.full != 0){
                for(cz = 0; cz < MIN(predictor_params.pred_bands + 3, z + 3); cz++){
                    bitStream_store(compressed_stream, written_bytes, written_bits, predictor_params.weight_init_resolution, predictor_params.weight_init_table[z][cz]);
                }
            }else{
                for(cz = 0; cz < MIN(predictor_params.pred_bands, z); cz++){
                    bitStream_store(compressed_stream, written_bytes, written_bits, predictor_params.weight_init_resolution, predictor_params.weight_init_table[z][cz]);
                }
            }
        }
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 8-(*written_bits), 0);
    }

    /* ENTROPY CODER METADATA */
    if(encoder_params.encoding_method == SAMPLE){
        // Unary length limit
        bitStream_store(compressed_stream, written_bytes, written_bits, 5, encoder_params.u_max);
        // rescaling counter size
        bitStream_store(compressed_stream, written_bytes, written_bits, 3, encoder_params.y_star - 4);
        // initial count exponent
        bitStream_store(compressed_stream, written_bytes, written_bits, 3, encoder_params.y_0);
        // Accumulator initialization constant and table
        if(encoder_params.k == (unsigned int)-1){
            unsigned int z = 0;
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 4, 1);
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
            for(z = 0; z < input_params.z_size; z++){
                bitStream_store(compressed_stream, written_bytes, written_bits, 4, encoder_params.k_init[z]);
            }
            if((input_params.z_size % 2) != 0)
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 4, 0);
        }else{
            bitStream_store(compressed_stream, written_bytes, written_bits, 4, encoder_params.k);
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
        }
    }else{
        // reserved
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
        // block size
        switch(encoder_params.block_size){
        case 8:
            bitStream_store(compressed_stream, written_bytes, written_bits, 2, 0x0);
            break;
        case 16:
            bitStream_store(compressed_stream, written_bytes, written_bits, 2, 0x1);
            break;
        case 32:
            bitStream_store(compressed_stream, written_bytes, written_bits, 2, 0x2);
            break;
        case 64:
            bitStream_store(compressed_stream, written_bytes, written_bits, 2, 0x3);
            break;
        }
        // Restricted code
        if(input_params.dyn_range <= 4 && encoder_params.restricted != 0)
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
        else
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
        // Reference Sample Interval
        bitStream_store(compressed_stream, written_bytes, written_bits, 12, encoder_params.ref_interval);
    }
}

__global__ void GPU_compute_k_split_enc(unsigned short *d_mpr, unsigned int * d_len, int * d_code){
    unsigned int x=(threadIdx.x*64)+blockIdx.x*4096;
    //if(blockIdx.x==0 && threadIdx.x==0)
    //printf("x=%u\n",x); 

    unsigned short mpr[64];
    unsigned int len[1];
    unsigned short code[1];

    //------------------------------------------------
    //leitura da imagem para shared memory
    for(int i=0; i<64; i++)
        mpr[i]=d_mpr[x+i];

    unsigned int code_len_temp = 0;
    
    len[0] = (unsigned int)-1;
    code[0]=0;
    for(int k = 0; k < 14; k++){
        code_len_temp = 0;
        for(int i=0; i < 64; i++){
            //if(threadIdx.x+blockIdx.x*64==4096 && k==0)
                //printf("code_len_temp=%u i=%d, mpr=%d\n", threadIdx.x, blockIdx.x, x); */
            //printf(" i=%d, k=%d, value=%hu, shift=%hu\n",i,k,args->ptr_k[i], (args->ptr_k[i] >> k));
            code_len_temp += (mpr[i] >> k) + 1 + k;
        }
        //if(threadIdx.x+blockIdx.x*64==4096)
        //printf("code_len_temp=%u\n", code_len_temp);

        if(code_len_temp < len[0]){
            
            len[0] = code_len_temp;
            code[0] = k;
            
        }
    }

    d_len[threadIdx.x+blockIdx.x*64]=len[0];
    d_code[threadIdx.x+blockIdx.x*64]=code[0];

}


__global__ void GPU_compute_sample(unsigned short *d_mpr, unsigned long * d_stream, unsigned int * d_bytes, unsigned int * d_bits, unsigned int dim_x, unsigned int dim_y){
    unsigned int z=threadIdx.x;


    unsigned int counter[1];
    unsigned int accumulator[1];
    int y=0;
    int x=0;

    counter[0]=0x1 << 1;
    accumulator[0]=(counter[0]*(3*(0x1 << (3 + 6))-49))/0x080;


        for(y = 0; y < dim_y; y++){
            for(x = 0; x < dim_x; x++){
                unsigned int curIndex = x + y*dim_x + z*dim_x*dim_y;
    
                if((y == 0 && x == 0)){
                    // I simply save on the output stream the unmodified
                    // residual (which should actually be the unmodified pixel)
                    //if(d_bytes[27]==0 && z==27)

                    //printf("zero x=%d, y=%d, counter=%u, accumulator=%u,  mpr=%hu, bits=%hu, value=%lu, result=%d\n", x, y, counter[0], accumulator[0], d_mpr[curIndex], d_bits[27], d_stream[(3*z)*(dim_x*dim_y)+0], (int)__log2f(((49*counter[0])/0x080 + accumulator[0])/((double)counter[0])));

                    gpu_bitStream_store(&d_stream[0], &d_bytes[0], &d_bits[0], 16, d_mpr[curIndex],(2*z)*(dim_x*dim_y),z);

                    //if(d_bytes[27]==0 && z==27)

                    //printf("zero x=%d, y=%d, counter=%u, accumulator=%u,  mpr=%hu, bits=%hu, value=%lu, result=%d\n", x, y, counter[0], accumulator[0], d_mpr[curIndex], d_bits[27], d_stream[(3*z)*(dim_x*dim_y)+0], (int)__log2f(((49*counter[0])/0x080 + accumulator[0])/((double)counter[0])));

                    
                    
                }
                else{
                    int temp_k = 0;
                    unsigned int divisor = 0;
                    unsigned int reminder = 0;
                    
                    // Now, general case, I have to actually perform the compression ...
                    temp_k = (int)__log2f(((49*counter[0])/0x080 + accumulator[0])/((double)counter[0]));
                    
                    if(temp_k < 0)
                        temp_k = 0;
                    if(temp_k > 14)
                        temp_k = 14;
                    divisor = d_mpr[curIndex]/(0x1 << temp_k);
                    reminder = d_mpr[curIndex] & (((unsigned short)0xFFFF) >> (16 - temp_k));

                   
                    // ... save the computation on the output stream ...
                    //if(d_bytes[27]==0 && z==27)

                    //printf("x=%d, y=%d, counter=%u, accumulator=%u, temp=%d, div=%u, rem=%u, mpr=%hu, bits=%hu, value=%lu, result=%d\n", x, y, counter[0], accumulator[0],temp_k, divisor, reminder, d_mpr[curIndex], d_bits[27], d_stream[(3*z)*(dim_x*dim_y)+0], (int)__log2f(((49*counter[0])/0x080 + accumulator[0])/((double)counter[0])));

                    
                    if(divisor < 8){
                        gpu_bitStream_store_constant(&d_stream[0], &d_bytes[0], &d_bits[0], divisor, 0,(2*z)*(dim_x*dim_y), z);
                        gpu_bitStream_store_constant(&d_stream[0], &d_bytes[0], &d_bits[0], 1, 1,(2*z)*(dim_x*dim_y), z);
                        gpu_bitStream_store(&d_stream[0], &d_bytes[0], &d_bits[0], temp_k, reminder,(2*z)*(dim_x*dim_y), z);
                    }
                    else{
                        gpu_bitStream_store_constant(&d_stream[0], &d_bytes[0], &d_bits[0], 8, 0,(2*z)*(dim_x*dim_y), z);
                        gpu_bitStream_store(&d_stream[0], &d_bytes[0], &d_bits[0], 16, d_mpr[curIndex],(2*z)*(dim_x*dim_y), z);
                    }
                    

                    //if(z==27 && x==4 && y==0)
                    //printf("x=%lu\n", d_stream[(3*z)*(dim_x*dim_y)]);

                    
                    // ... and finally update the statistics
                    if(counter[0] < ((((unsigned int)0x1) << 9) -1)){
                        accumulator[0] += d_mpr[curIndex];
                        counter[0]++;
                    }
                    else{
                        accumulator[0] = (accumulator[0] + d_mpr[curIndex] + 1)/2;
                        counter[0] = (counter[0] + 1)/2;
                    }
                }
            }
        }


}

void *compute_sample0(void * input){
    struct thread_args_stream_conc *args = (struct thread_args_stream_conc *) input;
    clock_gettime(CLOCK_MONOTONIC, &start2);

    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 8, 0);
    // x, y, z dimensions
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->input_params.x_size);
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->input_params.y_size);
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->input_params.z_size);
    // Sample type
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0);
    // dynamic range
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 16);
    // Encoding Sample Order and interleaving

    if(args->encoder_params.out_interleaving == BSQ){
        bitStream_store_constant(args->compressed_stream, args->written_bytes,args->written_bits, 1, 1);
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 16, 0);
    }
    else{
        bitStream_store_constant(args->compressed_stream, args->written_bytes,args-> written_bits, 1, 0);
        bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->encoder_params.out_interleaving_depth);
    }
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0);
    // Out word size
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 3, 1);
    // Encoder type
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 10, 0);

    /* PREDICTOR METADATA */
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0);
    // prediction bands
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, args->predictor_params.user_input_pred_bands);
    // prediction mode
    if(args->predictor_params.full != 0)
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    else
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // local sum
    if(args->predictor_params.neighbour_sum != 0)
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    else
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);

    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // Register size
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 6, 32);
    // Weight resolution
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 0);
    // weight update scaling exponent change interval
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, ((unsigned int)log2((float)2048)) - 4);
    // weight update scaling exponent initial parameter
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 0);
    // weight update scaling exponent final parameter
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 0);
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // weight initialization method and weight initialization table flag

    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0);
    // weight initialization resolution

    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 5, 0);
    // Weight initialization table


    /* ENTROPY CODER METADATA */

    
    /* ENTROPY CODER METADATA */
        // Unary length limit
        bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 5, 8);
        // rescaling counter size
        bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 3, 5);
        // initial count exponent
        bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 3, 8);
        // Accumulator initialization constant and table
        bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 14);
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
        

    //clock_gettime(CLOCK_MONOTONIC, &end2);
    //printf("BITSTREAM_HEADER=%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);
    

    //clock_gettime(CLOCK_MONOTONIC, &end2);
    //printf("BITSTREAM_SEM=%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes1[0], args->written_bits1[0]);


    int i=0;
    if(args->written_bits[0]==0){
        for(i=0; i<=args->written_bytes1[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream1[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes1[0];i++){
            //if(i==args->written_bytes1[0])
            //printf("aux=%u\n", args->written_bytes1[0]);
            args->compressed_stream[args->written_bytes[0]]|=(args->stream1[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream1[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits1[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream1[i]>>args->written_bits[0]);
            args->written_bytes[0]--;

        }
        /*else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream1[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream1[i]<<(args->written_bits1[0]-(args->written_bits1[0]+args->written_bits[0])%64));

        } */
    }
    args->written_bits[0]=(args->written_bits[0]+args->written_bits1[0])%64;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes2[0], args->written_bits2[0]);


    if(args->written_bits[0]==0){
        for(i=0; i<args->written_bytes2[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream2[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes2[0];i++){

            args->compressed_stream[args->written_bytes[0]]|=(args->stream2[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream2[i]<<(64-args->written_bits[0]));


        }
         if(args->written_bits[0]+args->written_bits2[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream2[i]>>args->written_bits[0]);
            args->written_bytes[0]--;

        }
        /*else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream2[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream2[i]<<(args->written_bits2[0]-(args->written_bits2[0]+args->written_bits[0])%64));

        } */
    }

    args->written_bits[0]=(args->written_bits[0]+args->written_bits2[0])%64;
    //args->written_bytes[0]--;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes3[0], args->written_bits3[0]);



    /*if(args->written_bits[0]==0){
        for(i=0; i<args->written_bytes3[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream3[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes3[0];i++){

            args->compressed_stream[args->written_bytes[0]]|=(args->stream3[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream3[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits3[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream3[i]>>args->written_bits[0]);
            args->written_bytes[0]--;
        }*/
        /* else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream3[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream3[i]<<(args->written_bits3[0]-(args->written_bits3[0]+args->written_bits[0])%64));

        } */
    /*}

    args->written_bits[0]=(args->written_bits[0]+args->written_bits3[0])%64;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes4[0], args->written_bits4[0]);



    if(args->written_bits[0]==0){
        for(i=0; i<args->written_bytes4[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream4[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes4[0];i++){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream4[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream4[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits4[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream4[i]>>args->written_bits[0]);
            args->written_bytes[0]--;
        }*/
        /* else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream4[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream4[i]<<(args->written_bits4[0]-(args->written_bits4[0]+args->written_bits[0])%64));

        } */
    /*} 

    args->written_bits[0]=(args->written_bits[0]+args->written_bits4[0])%64;*/
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes5[0], args->written_bits5[0]);


    /*if(args->written_bits[0]==0){
        for(i=0; i<args->written_bytes5[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream5[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes5[0];i++){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream5[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream5[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits5[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream5[i]>>args->written_bits[0]);
            args->written_bytes[0]--;
        }*/
        /* else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream5[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream5[i]<<(args->written_bits5[0]-(args->written_bits5[0]+args->written_bits[0])%64));

        } */
    /*}

    args->written_bits[0]=(args->written_bits[0]+args->written_bits5[0])%64;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes6[0], args->written_bits6[0]);



    if(args->written_bits[0]==0){
        for(i=0; i<=args->written_bytes6[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream6[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes6[0];i++){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream6[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream6[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits6[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream6[i]>>args->written_bits[0]);
            args->written_bytes[0]--;
        }

    }

    args->written_bits[0]=(args->written_bits[0]+args->written_bits6[0])%64;  
    //printf("byte=%u, bit=%u\n", args->written_bytes[0],args->written_bits[0]);*/


    //printf("bytes=%u, bits=%u\n", args->written_bytes[0],args->written_bits[0]);

    //printf("const_zero=%u, const_less=%u, const_diff_zero=%u, zero=%u, less=%u, diff_zero=%u, for1=%u, for2=%u, big_zero=%u\n", const_zero, const_less, const_diff_zero, zero, less, diff_zero, for1, for2, big_zero);
    unsigned char num_padding_bits=args->written_bits[0]%8;
    //printf("num_padding_bits=%hhu\n", num_padding_bits);
    if(num_padding_bits < 8 && num_padding_bits > 0){
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, num_padding_bits, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end2);
    printf("BITSTREAM=%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);

    FILE *fp;
    fp = fopen("tempos.txt","a");
    if(fp == NULL){
        printf("Error writng file!\n");   
        exit(1);             
    }
    fprintf(fp,"%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);
    fclose(fp);

    pthread_exit (NULL);

}

void *compute_sample(void *input){
    struct thread_args_sample *args = (struct thread_args_sample *) input;
    clock_gettime(CLOCK_MONOTONIC, &start2);

      // Let's remember that the elements are saved in residuals so that
    // element(x, y, z) = residuals[x + y*x_size + z*x_size*y_size], i.e.
    // they are saved in BSQ order

    unsigned int x = 0, y = 0, z = 0;

    for(z = 0; z < args->bands; z++){
        args->counter[z] = 0x1 << 8;
        args->accumulator[z] = (args->counter[z]*(3*(0x1 << (14 + 6))-49))/0x080;
        for(y = 0; y < args->input_params.y_size; y++){
            for(x = 0; x < args->input_params.x_size; x++){
                unsigned int curIndex = x + y*args->input_params.x_size + z*args->input_params.x_size*args->input_params.y_size;
    
                if((y == 0 && x == 0)){
                    // I simply save on the output stream the unmodified
                    // residual (which should actually be the unmodified pixel)
                    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->residuals[curIndex]);
                }
                else{
                    int temp_k = 0;
                    unsigned int divisor = 0;
                    unsigned int reminder = 0;
                    // Now, general case, I have to actually perform the compression ...
                    temp_k = (int)log2(((49*args->counter[z])/0x080 + args->accumulator[z])/((double)args->counter[z]));
                    if(temp_k < 0)
                        temp_k = 0;
                    if(temp_k > 14)
                        temp_k = 14;
                    divisor = args->residuals[curIndex]/(0x1 << temp_k);
                    reminder = args->residuals[curIndex] & (((unsigned short)0xFFFF) >> (16 - temp_k));

                    // ... save the computation on the output stream ...
                    if(divisor < 8){
                        //if(args->written_bytes[0]==198752/8 || args->written_bytes[0]==198744/8)
                        //printf("1stream=%x, bits=%d, divisor=%u\n",args->compressed_stream[args->written_bytes[0]],args->written_bits[0], divisor);
                        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, divisor, 0);
                        //if(args->written_bytes[0]==198752/8 || args->written_bytes[0]==198744/8)
                        //printf("2stream=%x, bits=%d\n",args->compressed_stream[args->written_bytes[0]],args->written_bits[0]);
                        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                        //if(args->written_bytes[0]==198752/8 || args->written_bytes[0]==198744/8)
                        //printf("3stream=%x, bits=%d, k=%d, remaider=%u\n",args->compressed_stream[args->written_bytes[0]],args->written_bits[0], temp_k, reminder);
                        bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, temp_k, reminder);
                        //if(args->written_bytes[0]==198752/8 || args->written_bytes[0]==198744/8)
                        //printf("4stream=%x, bits=%d\n",args->compressed_stream[args->written_bytes[0]],args->written_bits[0]);
                    }
                    else{
                        //if(args->written_bytes[0]==198752/8 || args->written_bytes[0]==198744/8)
                        //printf("5stream=%x, bits=%d\n",args->compressed_stream[args->written_bytes[0]],args->written_bits[0]);
                        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 8, 0);
                        //if(args->written_bytes[0]==198752/8 || args->written_bytes[0]==198744/8)
                        //printf("6stream=%x, bits=%d, residual=%d\n",args->compressed_stream[args->written_bytes[0]],args->written_bits[0], args->residuals[curIndex]);
                        bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->residuals[curIndex]);
                        //if(args->written_bytes[0]==198752/8 || args->written_bytes[0]==198744/8)
                        //printf("7stream=%x, bits=%d\n",args->compressed_stream[args->written_bytes[0]],args->written_bits[0]);
                    }

                    
                    // ... and finally update the statistics
                    if(args->counter[z] < ((((unsigned int)0x1) << 9) -1)){
                        args->accumulator[z] += args->residuals[curIndex];
                        args->counter[z]++;
                    }
                    else{
                        args->accumulator[z] = (args->accumulator[z] + args->residuals[curIndex] + 1)/2;
                        args->counter[z] = (args->counter[z] + 1)/2;
                    }
                }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end2);
    printf("core=%d, BITSTREAM=%lf\n",args->bands, (end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6); 

   
    FILE *fp;
    fp = fopen("tempos_arm.txt","a");
    if(fp == NULL){
        printf("Error writng file!\n");   
        exit(1);             
    }
    fprintf(fp,"%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6 );
    fclose(fp); 
    

    
    
    pthread_exit (NULL);

}

/// Given a single residual and the statistics accumulated so far, it computes the code
/// for the residual and it updates the statistics.
extern "C" int encode_pixel(unsigned int x, unsigned int y, unsigned int z, unsigned int * counter, unsigned int * accumulator,
    unsigned int *written_bytes, unsigned int * written_bits, unsigned long * compressed_stream, unsigned short int * residuals,
    input_feature_t input_params, encoder_config_t encoder_params){
    unsigned int curIndex = x + y*input_params.x_size + z*input_params.x_size*input_params.y_size;
    
    if((y == 0 && x == 0)){
        // I simply save on the output stream the unmodified
        // residual (which should actually be the unmodified pixel)
        bitStream_store(compressed_stream, written_bytes, written_bits, input_params.dyn_range, residuals[curIndex]);
    }
    else{
        int temp_k = 0;
        unsigned int divisor = 0;
        unsigned int reminder = 0;
        // Now, general case, I have to actually perform the compression ...
        temp_k = (int)log2(((49*counter[z])/0x080 + accumulator[z])/((double)counter[z]));
        if(temp_k < 0)
            temp_k = 0;
        if(temp_k > (input_params.dyn_range - 2))
            temp_k = input_params.dyn_range - 2;
        divisor = residuals[curIndex]/(0x1 << temp_k);
        reminder = residuals[curIndex] & (((unsigned short)0xFFFF) >> (16 - temp_k));

        // ... save the computation on the output stream ...
        if(divisor < encoder_params.u_max){
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, divisor, 0);
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
            bitStream_store(compressed_stream, written_bytes, written_bits, temp_k, reminder);
        }
        else{
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, encoder_params.u_max, 0);
            bitStream_store(compressed_stream, written_bytes, written_bits, input_params.dyn_range, residuals[curIndex]);
        }
        
        // ... and finally update the statistics
        if(counter[z] < ((((unsigned int)0x1) << encoder_params.y_star) -1)){
            accumulator[z] += residuals[curIndex];
            counter[z]++;
        }
        else{
            accumulator[z] = (accumulator[z] + residuals[curIndex] + 1)/2;
            counter[z] = (counter[z] + 1)/2;
        }
    }
    
//#ifndef NDEBUG
    if(*written_bytes > input_params.dyn_range*input_params.x_size*input_params.y_size*input_params.z_size){
        fprintf(stderr, "Error in encode_pixel, writing outside the compressed_stream boundaries: it means that the compressed image is greater than the original\n");
        return -1;
    }
//#endif
    return 0;
}

///Given the characteristics of the input stream, the parameters describing the desired behavior
///of the encoder and the list of residuals to be encoded (note that each residual is treated as
///an integer) it returs the size in bytes of the stream containing the compressed residuals (saved into compressed_stream)
///After usage, the caller has to deallocate the memory area pointed by compressed_stream and allocated by this function.
///@param input_params describe the image whose residuals are contained in the input file
///@param encoder_params set of options determining the behavior of the encoder
///@param residuals array containing the information to be compressed
///@param compressed_stream pointer to the array which, at the end of this function, will contain the compressed information
///the array is allocated inside this function
///@return a negative number if an error occurred
extern "C" int encode_sampleadaptive(input_feature_t input_params,predictor_config_t predictor_params, encoder_config_t encoder_params, unsigned short int * residuals,
    unsigned long * compressed_stream, unsigned int * written_bytes, unsigned int * written_bits){
    //First of all we proceed with the compression of the residuals according to the
    //sample adaptive encodying method, as specified in the header of this file.
    //For simplicity I proceed with encoding in the order into which the encoded samples
    //have to be saved into the output stream. Statistics (counter and accumulator) are anyway maintained per band,
    //so, even if samples from different bands are interleaved on the output stream,
    //it is as if each band were encoded separately.





//printf("%llu\n", millisecondsSinceEpoch);

    if(mlockall(MCL_CURRENT|MCL_FUTURE) == -1)
        printf("mlockall\n");

    args_sample1 = (struct thread_args_sample *)malloc(sizeof(struct thread_args_sample));
    args_sample2 = (struct thread_args_sample *)malloc(sizeof(struct thread_args_sample));
    args_sample3 = (struct thread_args_sample *)malloc(sizeof(struct thread_args_sample));
    args_sample4 = (struct thread_args_sample *)malloc(sizeof(struct thread_args_sample));
    //args_sample5 = (struct thread_args_sample *)malloc(sizeof(struct thread_args_sample));
    //args_sample6 = (struct thread_args_sample *)malloc(sizeof(struct thread_args_sample));
    args_sample0 = (struct thread_args_stream_conc *)malloc(sizeof(struct thread_args_stream_conc));


    unsigned long * stream1=NULL;
    unsigned long * stream2=NULL;
    unsigned long * stream3=NULL;
    unsigned long * stream4=NULL;
    //unsigned long * stream5=NULL;
    //unsigned long * stream6=NULL;


    unsigned int * written_bytes1=NULL;
    unsigned int * written_bits1=NULL;
    unsigned int * written_bytes2=NULL;
    unsigned int * written_bits2=NULL;
    unsigned int * written_bytes3=NULL;
    unsigned int * written_bits3=NULL;
    unsigned int * written_bytes4=NULL;
    unsigned int * written_bits4=NULL;
    //unsigned int * written_bytes5=NULL;
    //unsigned int * written_bits5=NULL;
    //unsigned int * written_bytes6=NULL;
    //unsigned int * written_bits6=NULL;


    unsigned int * counter1 = NULL;
    unsigned int * accumulator1 = NULL;
    unsigned int * counter2 = NULL;
    unsigned int * accumulator2 = NULL;
    unsigned int * counter3 = NULL;
    unsigned int * accumulator3 = NULL;
    unsigned int * counter4 = NULL;
    unsigned int * accumulator4 = NULL;
    //unsigned int * counter5 = NULL;
    //unsigned int * accumulator5 = NULL;
    //unsigned int * counter6 = NULL;
    //unsigned int * accumulator6 = NULL;

     //evita que a memoria seja usada para swap
     //unsigned int arm=(((input_params.z_size)*20)/100)+1;
     //unsigned int denver=(((input_params.z_size)*10)/100);
     //unsigned int denver=48;
    /* #ifdef CRISM
        unsigned int arm=(input_params.z_size-1)/4;
    #else 
        unsigned int arm=input_params.z_size/4;
    #endif */

    #ifdef CRISM
        unsigned int arm=(input_params.z_size-1)/2;
    #else 
        unsigned int arm=input_params.z_size/2;
    #endif
     
     
     //arm=arm+((rem_arm+rem_denver)/10)/4;
     //denver++;
     //printf("arm=%u denver=%u, rem arm=%u, rem denver=%u\n", arm, denver, rem_arm, rem_denver);


    
    stream1=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream1 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream1, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    stream2=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream2 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream2, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    stream3=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream3, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    stream4=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream4, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));


    /*stream5=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream5, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    stream6=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream6, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));*/

    //==============================================================================================================================
    //stream 1
    written_bytes1=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes1 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits1=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits1 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes1[0]=0;
    written_bits1[0]=0;


    //==============================================================================================================================
    //stream 2

        written_bytes2=(unsigned int *)malloc(sizeof(unsigned int));
        if(written_bytes2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        written_bits2=(unsigned int *)malloc(sizeof(unsigned int));
        if(written_bits2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        written_bytes2[0]=0;
        written_bits2[0]=0;

    //==============================================================================================================================
    //stream 3
    written_bytes3=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits3=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes3[0]=0;
    written_bits3[0]=0;

    //==============================================================================================================================
    //stream 4
    written_bytes4=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits4=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes4[0]=0;
    written_bits4[0]=0;

    //==============================================================================================================================
    //stream 5
    /*written_bytes5=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits5=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes5[0]=0;
    written_bits5[0]=0;

    //==============================================================================================================================
    //stream 6
    written_bytes6=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits6=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes6[0]=0;
    written_bits6[0]=0;*/



    counter1 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(counter1 == NULL){
        fprintf(stderr, "Error in the allocation of the counter2 statistic\n\n");
        return -1;
    }
    accumulator1 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(accumulator1 == NULL){
        fprintf(stderr, "Error in the allocation of the accumulator1 statistic\n\n");
        return -1;
    }

    /* counter2 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(counter2 == NULL){
        fprintf(stderr, "Error in the allocation of the counter2 statistic\n\n");
        return -1;
    }
    accumulator2 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(accumulator2 == NULL){
        fprintf(stderr, "Error in the allocation of the accumulator2 statistic\n\n");
        return -1;
    }

    counter3 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(counter3 == NULL){
        fprintf(stderr, "Error in the allocation of the counter3 statistic\n\n");
        return -1;
    }
    accumulator3 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(accumulator3 == NULL){
        fprintf(stderr, "Error in the allocation of the accumulator3 statistic\n\n");
        return -1;
    } */

    /* #ifdef CRISM
        counter4 = (unsigned int *)malloc(sizeof(unsigned int)*(arm+1));
        if(counter4 == NULL){
            fprintf(stderr, "Error in the allocation of the counter4 statistic\n\n");
            return -1;
        }
        accumulator4 = (unsigned int *)malloc(sizeof(unsigned int)*(arm+1));
        if(accumulator4 == NULL){
            fprintf(stderr, "Error in the allocation of the accumulator4 statistic\n\n");
            return -1;
        }
    #else 
        counter4 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
        if(counter4 == NULL){
            fprintf(stderr, "Error in the allocation of the counter4 statistic\n\n");
            return -1;
        }
        accumulator4 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
        if(accumulator4 == NULL){
            fprintf(stderr, "Error in the allocation of the accumulator4 statistic\n\n");
            return -1;
        }
    #endif */

    #ifdef CRISM
        counter2 = (unsigned int *)malloc(sizeof(unsigned int)*(arm+1));
        if(counter2 == NULL){
            fprintf(stderr, "Error in the allocation of the counter4 statistic\n\n");
            return -1;
        }
        accumulator2 = (unsigned int *)malloc(sizeof(unsigned int)*(arm+1));
        if(accumulator2 == NULL){
            fprintf(stderr, "Error in the allocation of the accumulator4 statistic\n\n");
            return -1;
        }
    #else 
        counter2 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
        if(counter2 == NULL){
            fprintf(stderr, "Error in the allocation of the counter4 statistic\n\n");
            return -1;
        }
        accumulator2 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
        if(accumulator2 == NULL){
            fprintf(stderr, "Error in the allocation of the accumulator4 statistic\n\n");
            return -1;
        }
    #endif


    /*counter5 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(counter5 == NULL){
        fprintf(stderr, "Error in the allocation of the counter5 statistic\n\n");
        return -1;
    }
    accumulator5 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(accumulator5 == NULL){
        fprintf(stderr, "Error in the allocation of the accumulator5 statistic\n\n");
        return -1;
    }

    counter6 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(counter6 == NULL){
        fprintf(stderr, "Error in the allocation of the counter6 statistic\n\n");
        return -1;
    }
    accumulator6 = (unsigned int *)malloc(sizeof(unsigned int)*arm);
    if(accumulator6 == NULL){
        fprintf(stderr, "Error in the allocation of the accumulator6 statistic\n\n");
        return -1;
    }*/

       

    args_sample1->compressed_stream=&stream1[0];
    args_sample2->compressed_stream=&stream2[0];
    args_sample3->compressed_stream=&stream3[0];
    args_sample4->compressed_stream=&stream4[0];
    //args_sample5->compressed_stream=&stream5[0];
    //args_sample6->compressed_stream=&stream6[0];

    args_sample1->written_bytes=&written_bytes1[0];
    args_sample1->written_bits=&written_bits1[0];
    args_sample2->written_bytes=&written_bytes2[0];
    args_sample2->written_bits=&written_bits2[0];
    args_sample3->written_bytes=&written_bytes3[0];
    args_sample3->written_bits=&written_bits3[0];
    args_sample4->written_bytes=&written_bytes4[0];
    args_sample4->written_bits=&written_bits4[0];
    /*args_sample5->written_bytes=&written_bytes5[0];
    args_sample5->written_bits=&written_bits5[0];
    args_sample6->written_bytes=&written_bytes6[0];
    args_sample6->written_bits=&written_bits6[0];*/

    args_sample1->residuals=&residuals[0];
    args_sample2->residuals=&residuals[arm*input_params.y_size*input_params.x_size];
    args_sample3->residuals=&residuals[(2*arm)*input_params.y_size*input_params.x_size];
    args_sample4->residuals=&residuals[(3*arm)*input_params.y_size*input_params.x_size];
    //args_sample5->residuals=&residuals[((2*denver)+(2*arm))*input_params.y_size*input_params.x_size];
    //args_sample6->residuals=&residuals[((2*denver)+(3*arm))*input_params.y_size*input_params.x_size];

    args_sample1->counter=&counter1[0];
    args_sample2->counter=&counter2[0];
    args_sample3->counter=&counter3[0];
    args_sample4->counter=&counter4[0];
    //args_sample5->counter=&counter5[0];
    //args_sample6->counter=&counter6[0];

    args_sample1->accumulator=&accumulator1[0];
    args_sample2->accumulator=&accumulator2[0];
    args_sample3->accumulator=&accumulator3[0];
    args_sample4->accumulator=&accumulator4[0];
    //args_sample5->accumulator=&accumulator5[0];
    //args_sample6->accumulator=&accumulator6[0];


    args_sample1->bands=arm;
    args_sample2->bands=arm;
    args_sample3->bands=arm;

     /* #ifdef CRISM
        args_sample4->bands=arm+1;
    #else
        args_sample4->bands=arm;
    #endif  */
   //args_sample5->bands=arm;
    //args_sample6->bands=arm;

    #ifdef CRISM
        args_sample2->bands=arm+1;
    #else
        args_sample2->bands=arm;
    #endif 
    

    args_sample1->input_params=input_params;
    args_sample2->input_params=input_params;
    args_sample3->input_params=input_params;
    args_sample4->input_params=input_params;
    //args_sample5->input_params=input_params;
    //args_sample6->input_params=input_params;


    args_sample0->compressed_stream=&compressed_stream[0];
    args_sample0->written_bytes=&written_bytes[0];
    args_sample0->written_bits=&written_bits[0];
    args_sample0->stream1=&stream1[0];
    args_sample0->written_bytes1=&written_bytes1[0];
    args_sample0->written_bits1=&written_bits1[0];
    args_sample0->stream2=&stream2[0];
    args_sample0->written_bytes2=&written_bytes2[0];
    args_sample0->written_bits2=&written_bits2[0];
    args_sample0->stream3=&stream3[0];
    args_sample0->written_bytes3=&written_bytes3[0];
    args_sample0->written_bits3=&written_bits3[0];
    args_sample0->stream4=&stream4[0];
    args_sample0->written_bytes4=&written_bytes4[0];
    args_sample0->written_bits4=&written_bits4[0];
    /*args_sample0->stream5=&stream5[0];
    args_sample0->written_bytes5=&written_bytes5[0];
    args_sample0->written_bits5=&written_bits5[0];
    args_sample0->stream6=&stream6[0];
    args_sample0->written_bytes6=&written_bytes6[0];
    args_sample0->written_bits6=&written_bits6[0];*/
    args_sample0->input_params=input_params;
    args_sample0->predictor_params=predictor_params;
    args_sample0->encoder_params=encoder_params;


    


    
    CPU_ZERO(&cpuset_stream1);
    CPU_SET(0, &cpuset_stream1);

    
    CPU_ZERO(&cpuset_stream2);
    CPU_SET(1, &cpuset_stream2);


    CPU_ZERO(&cpuset_stream3);
    CPU_SET(2, &cpuset_stream3);

    CPU_ZERO(&cpuset_stream4);
    CPU_SET(3, &cpuset_stream4);

    /*CPU_ZERO(&cpuset_stream5);
    CPU_SET(4, &cpuset_stream5);
    
    CPU_ZERO(&cpuset_stream6);
    CPU_SET(5, &cpuset_stream6);*/


    CPU_ZERO(&cpuset_stream0);
    CPU_SET(1, &cpuset_stream0);
    CPU_SET(0, &cpuset_stream0);
    /* CPU_SET(0, &cpuset_stream0);
    CPU_SET(3, &cpuset_stream0);
 */
    printf("start\n");



    if(pthread_attr_init(&(threads_attr[1])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[1]), sizeof(cpu_set_t), &cpuset_stream1) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[2])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[2]), sizeof(cpu_set_t), &cpuset_stream2) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[3])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[3]), sizeof(cpu_set_t), &cpuset_stream3) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[4])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[4]), sizeof(cpu_set_t), &cpuset_stream4) != 0)
        printf("pthread_attr_setaffinity_np");

    /*if(pthread_attr_init(&(threads_attr[5])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[5]), sizeof(cpu_set_t), &cpuset_stream5) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[6])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[6]), sizeof(cpu_set_t), &cpuset_stream6) != 0)
        printf("pthread_attr_setaffinity_np");*/

        if(pthread_attr_init(&(threads_attr[0])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[0]), sizeof(cpu_set_t), &cpuset_stream0) != 0)
        printf("pthread_attr_setaffinity_np");


        /* struct timeval tv1, tv2;

        gettimeofday(&tv1, NULL); */
    //clock_gettime(CLOCK_MONOTONIC, &start1);
    
    
    pthread_create(&threads[1], &(threads_attr[1]), compute_sample, args_sample1);
    pthread_create(&threads[2], &(threads_attr[2]), compute_sample, args_sample2);
    /* pthread_create(&threads[3], &(threads_attr[3]), compute_sample, args_sample3);
    pthread_create(&threads[4], &(threads_attr[4]), compute_sample, args_sample4); */
    //pthread_create(&threads[5], &(threads_attr[5]), compute_sample, args_sample5);
    //pthread_create(&threads[6], &(threads_attr[6]), compute_sample, args_sample6);
    
    pthread_join(threads[1], NULL);
    pthread_join(threads[2], NULL);
    /* pthread_join(threads[3], NULL);
    pthread_join(threads[4], NULL); */
    //pthread_join(threads[5], NULL);
    //pthread_join(threads[6], NULL);


    pthread_create(&threads[0], &(threads_attr[0]), compute_sample0, args_sample0);
    pthread_join(threads[0], NULL); 
    
    /*clock_gettime(CLOCK_MONOTONIC, &end1);
    printf("total BITSTREAM=%lf\n", (end1.tv_sec-start1.tv_sec)*1e3+(end1.tv_nsec-start1.tv_nsec)*1e-6);

    FILE *fp;
    fp = fopen("tempos.txt","a");
    if(fp == NULL){
        printf("Error writng file!\n");   
        exit(1);             
    }
    fprintf(fp,"%lf\n",(end1.tv_sec-start1.tv_sec)*1e3+(end1.tv_nsec-start1.tv_nsec)*1e-6);
    fclose(fp);*/ 

        /*gettimeofday(&tv2, NULL);

    /* unsigned long long millisecondsSinceEpoch =
    (unsigned long long)(tv1.tv_sec) * 1000 +
    (unsigned long long)(tv1.tv_usec) / 1000;

    FILE *fp;
    fp = fopen("tempos.txt","a");
    if(fp == NULL){
        printf("Error writng file!\n");   
        exit(1);             
    }
    fprintf(fp,"%llu\n", millisecondsSinceEpoch);
    millisecondsSinceEpoch =
    (unsigned long long)(tv2.tv_sec) * 1000 +
    (unsigned long long)(tv2.tv_usec) / 1000;
    fprintf(fp,"%llu\n", millisecondsSinceEpoch);
    fclose(fp); */
  


    free(args_sample1);
    free(args_sample2);
    free(args_sample3);
    free(args_sample4);
    //free(args_sample5);
    //free(args_sample6);

    free(written_bytes1);
    free(written_bytes2);
    free(written_bytes3);
    free(written_bytes4);
    //free(written_bytes5);
    //free(written_bytes6);

    free(written_bits1);
    free(written_bits2);
    free(written_bits3);
    free(written_bits4);
    //free(written_bits5);
    //free(written_bits6);

    free(stream1);
    free(stream2);
    free(stream3);
    free(stream4);
    //free(stream5);
    //free(stream6);

    free(counter1);
    free(counter2);
    free(counter3);
    free(counter4);
    //free(counter5);
    //free(counter6);

    free(accumulator1);
    free(accumulator2);
    free(accumulator3);
    free(accumulator4);
    //free(accumulator5);
    //free(accumulator6);

    munlockall();//desbloqueia memoria

    return 0;
}

/******************************************************
* END Sample Adaptive Routines
*******************************************************/

/******************************************************
* Routines for the Block Adaptive Encoder
*******************************************************/

/// This procedure simply computes the code corresponding to the num_zero_blocks number
/// of sequential blocks whose samples are all 0.
/// Such code is saved in the output bitstream.
extern "C" void zero_block_code(input_feature_t input_params, encoder_config_t encoder_params,
    int num_zero_blocks, unsigned long * compressed_stream, unsigned int * written_bytes, unsigned int * written_bits, int end_of_segment){
    //First of all I have to save the ID of the zero block code option
    printf("end segment\n");
    if(input_params.dyn_range <= 4 && encoder_params.restricted != 0){
        if(input_params.dyn_range < 3)
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
        else
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 3, 0);
    }else{
        if(input_params.dyn_range <= 8){
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 4, 0);
        }else if(input_params.dyn_range <= 16){
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 5, 0);
        }else{
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 6, 0);
        }
    }
    //Now I can compute and save the code indicating the number of zero blocks.
    if(num_zero_blocks > 1){
        if(num_zero_blocks < 5){
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, num_zero_blocks - 1, 0);
        }else{
            if(end_of_segment != 0)
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 4, 0);
            else
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, num_zero_blocks, 0);
        }
    }
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
}

/// Computes the values for the second extension compression option and the length
/// of the compression considering such option
//extern "C" unsigned int compute_second_extension(encoder_config_t encoder_params, unsigned short int * block_samples, unsigned int second_extension_values[32]){
void *compute_second_extension(void *input){
    struct thread_args_second *args = (struct thread_args_second *) input;
    unsigned int code_len = 0;
    int i = 0;
    //printf("arg=%d\n", args->return_arg);
    //#pragma omp parallel
    //{
        //#pragma omp for reduction(+:code_len) 
        
        for(i = 0; i < args->encoder_params.block_size; i+=2){

            args->second_extension_values[i/2] = (((unsigned long long)args->block_samples[i] + args->block_samples[i + 1])*((unsigned long long)args->block_samples[i] + args->block_samples[i + 1] + 1))/2 + args->block_samples[i + 1];
            code_len += args->second_extension_values[i/2] + 1;
            //if(args->second_extension_values[i/2]!=aux[args->return_arg*32+(i/2)] )
                //printf("original=%u, thread=%u\n",args->second_extension_values[i/2], aux[args->return_arg*32+(i/2)]  );
        }
    //}
    
    //if(code_len!=aux[args->return_arg] )
        //printf("original=%u, thread=%u, k=%d\n",code_len, aux[args->return_arg],args->return_arg  );
    args->return_arg=code_len;
    
    //return code_len;
    
    pthread_exit (NULL);
}

/// The length of the compression considering the bes k-split option
//extern "C" unsigned int compute_ksplit(input_feature_t input_params, encoder_config_t encoder_params, unsigned short int * block_samples, int * k_split, int dump){
    void *compute_ksplit(void * input){
        struct thread_args_split *args = (struct thread_args_split *) input;
        unsigned int code_len = (unsigned int)-1;
        int i = 0, k = 0;
        int k_limit = 0;
        if(args->input_params.dyn_range == 16){
            k_limit = 14;
        }else if(args->input_params.dyn_range == 8){
            k_limit = 6;
        }else if(args->input_params.dyn_range == 4 && args->encoder_params.restricted != 0){
            k_limit = 2;
        }else{
            k_limit = args->input_params.dyn_range;
        }
        
            for(k = 0; k < k_limit; k++){
                unsigned int code_len_temp = 0;
                //#pragma omp parallel
                //{
                    //#pragma omp for reduction(+:code_len)
                    for(i = 0; i < args->encoder_params.block_size; i++){
                        //printf("num %d\n",omp_get_thread_num());
                        code_len_temp += (args->block_samples[i] >> k) + 1 + k;
                    }
                //}
                if(code_len_temp < code_len){
                    code_len = code_len_temp;
                    *args->k_split = k;
                }
            }
            //exit(0);
        //return code_len;
        /*if(aux[args->return_arg]!=args->k_split[0])
            printf("original=%u, thread=%u, k=%u\n", args->k_split[0], aux[args->return_arg], args->return_arg);*/
        //if(aux[args->return_arg]!=code_len)
            //printf("original=%u, thread=%u, k=%u\n", code_len, aux[args->return_arg], args->return_arg);
        args->return_arg=code_len;
        pthread_exit (NULL);
    
    }

void *compute_zero_code(void * input){
    struct thread_args_zero *args = (struct thread_args_zero *) input;
    //clock_gettime(CLOCK_MONOTONIC, &start5);
    int i = 0, k=0;
    while(i < args->size){
        if(args->ptr_zero_block[i]!=0){
            args->zero_codes[k]=0;
            //printf("post zero");
            //sem_post(&sem_zero);
            k++;
            i=k*64;
        }
        else{
            i++;
            if(i%64==0){
                args->zero_codes[k]=1;
                //printf("post zero");
                //sem_post(&sem_zero);
                k++;
            }
        }
    }
    //clock_gettime(CLOCK_MONOTONIC, &end5);
    //printf("ZERO=%lf\n",(end5.tv_sec-start5.tv_sec)*1e3+(end5.tv_nsec-start5.tv_nsec)*1e-6);
    pthread_exit (NULL);
}

/// Computes the values for the second extension compression option and the length
/// of the compression considering such option
void *compute_sec_ext(void *input){
    struct thread_args_sec_ext *args = (struct thread_args_sec_ext *) input;
    //clock_gettime(CLOCK_MONOTONIC, &start4);
    unsigned int code_len = 0;
    int i = 0, k=0;

        for(i = 0; i < args->size; i+=2){
            if(i%64==0 && i!=0){
                args->len_sec_ext[k]=code_len;
                //printf("post sec");
                sem_post(&sem_sec_ext);
                k++;
                code_len=0;
            }
            //if(i<64)
                //printf("value1=%u, value2=%u, code_len=%u aux=%u\n",args->ptr_sec_ext[i], args->ptr_sec_ext[i+1], code_len, (((unsigned int)args->ptr_sec_ext[i] + args->ptr_sec_ext[i + 1])*((unsigned int)args->ptr_sec_ext[i] + args->ptr_sec_ext[i + 1] + 1))/2 + args->ptr_sec_ext[i + 1] );
            args->sec_ext_codes[i/2] = (((unsigned int)args->ptr_sec_ext[i] + args->ptr_sec_ext[i + 1])*((unsigned int)args->ptr_sec_ext[i] + args->ptr_sec_ext[i + 1] + 1))/2 + args->ptr_sec_ext[i + 1];
            code_len += args->sec_ext_codes[i/2] + 1;
            
        }
        //sem_post(&sem_sec_ext);
        //clock_gettime(CLOCK_MONOTONIC, &end4);
        //printf("SEC=%lf\n",(end4.tv_sec-start4.tv_sec)*1e3+(end4.tv_nsec-start4.tv_nsec)*1e-6);
    pthread_exit (NULL);
}



void *compute_k(void * input){
    struct thread_args_k *args = (struct thread_args_k *) input;
    //clock_gettime(CLOCK_MONOTONIC, &start3);


    cudaError_t err=cudaSuccess;


    dim3 enc_threadsPerBlock(64);
    dim3 enc_numBlocks(((args->size)/64)/64);

    size_t size_len=sizeof(unsigned int)*((args->size)/64);
    size_t size_codes=(sizeof(int)*((args->size)/64));
    size_t size_omega=sizeof(unsigned short)*args->size;

    unsigned int *d_len=NULL;
    int *d_code=NULL;
    unsigned short *d_mpr=NULL;

    err=cudaMalloc((void **)&d_len, size_len);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate device samples (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }
    err=cudaMalloc((void **)&d_code, size_codes);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate device samples (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }
    err=cudaMalloc((void **)&d_mpr, size_omega);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate device samples (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }


    err=cudaMemcpy(d_mpr, args->ptr_k, size_omega, cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to copy the samples from host to device (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    } 

    GPU_compute_k_split_enc<<<enc_numBlocks, enc_threadsPerBlock>>>(d_mpr, d_len, d_code);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to launch the kernel for the calculation of the central local difference (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }



    err=cudaMemcpy(args->len_k , d_len, size_len, cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to copy the mpr from device to host (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err=cudaMemcpy(args->k_codes , d_code, size_codes, cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to copy the mpr from device to host (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }


        //sem_post(&sem_k); 
    

    err=cudaFree(d_len);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to free the mpr from the device (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err=cudaFree(d_code);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to free the mpr from the device (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err=cudaFree(d_mpr);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to free the mpr from the device (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }
    //clock_gettime(CLOCK_MONOTONIC, &end3);
    //printf("K=%lf\n",(end3.tv_sec-start3.tv_sec)*1e3+(end3.tv_nsec-start3.tv_nsec)*1e-6);
    pthread_exit (NULL);



    /* cudaError_t err=cudaSuccess;

    int nStreams=1075;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i){
        err=cudaStreamCreate(&stream[i]);
        if(err!=cudaSuccess){
            fprintf(stderr, "Failed to create streams(error code %d)!\n", cudaGetLastError());
            exit(EXIT_FAILURE);
        }
    }

    dim3 enc_threadsPerBlock(1024);
    dim3 enc_numBlocks(1);

    size_t size_len=sizeof(unsigned int)*(1100800);
    size_t size_codes=(sizeof(int)*(1100800));
    size_t size_omega=sizeof(unsigned short)*70451200;

    size_t streamBytes_in=sizeof(unsigned short)*65536;
    size_t streamBytes_out=sizeof(unsigned int)*1024;
    size_t streamBytes_out_codes=sizeof(int)*1024;

    unsigned int *d_len=NULL;
    int *d_code=NULL;
    unsigned short *d_mpr=NULL;

    err=cudaMalloc((void **)&d_len, size_len);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate device samples (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }
    err=cudaMalloc((void **)&d_code, size_codes);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate device samples (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }
    err=cudaMalloc((void **)&d_mpr, size_omega);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate device samples (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * 65536;
        err=cudaMemcpyAsync(&d_mpr[offset], &args->ptr_k[offset], streamBytes_in, cudaMemcpyHostToDevice, stream[i]);
        if(err!=cudaSuccess){
            fprintf(stderr, "Failed to copy the image samples from host to device (error code %d)!\n", cudaGetLastError());
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < nStreams; ++i) {
        GPU_compute_k_split_enc<<<enc_numBlocks, enc_threadsPerBlock,0, stream[i]>>>(d_mpr, d_len, d_code, i);
        if(err!=cudaSuccess){
            fprintf(stderr, "Failed to launch the kernel for the calculation of the local sum (error code %d)!\n", cudaGetLastError());
            exit(EXIT_FAILURE);
        }
    }


    for (int i = 0; i < nStreams; ++i) {
        int offset = i * 1024;
        err=cudaMemcpyAsync(&args->len_k[offset], &d_len[offset], streamBytes_out, cudaMemcpyDeviceToHost, stream[i]);
        if(err!=cudaSuccess){
            fprintf(stderr, "Failed to copy the scaled from device to host (error code %d) id=%d!\n", cudaGetLastError(),i);
            exit(EXIT_FAILURE);
        }

        err=cudaMemcpyAsync(&args->k_codes[offset], &d_code[offset], streamBytes_out_codes, cudaMemcpyDeviceToHost, stream[i]);
        if(err!=cudaSuccess){
            fprintf(stderr, "Failed to copy the scaled from device to host (error code %d)!\n", cudaGetLastError());
            exit(EXIT_FAILURE);
        }

        //for(int j=0; j<512; j++){
            sem_post(&sem_k);
        //}


    }

    

    err=cudaFree(d_len);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to free the mpr from the device (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err=cudaFree(d_code);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to free the mpr from the device (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err=cudaFree(d_mpr);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to free the mpr from the device (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }
    //clock_gettime(CLOCK_MONOTONIC, &end3);
    //printf("K=%lf\n",(end3.tv_sec-start3.tv_sec)*1e3+(end3.tv_nsec-start3.tv_nsec)*1e-6);
    pthread_exit (NULL); */



   /*  unsigned int code_len = (unsigned int)-1;
    unsigned int code_len_temp = 0;
    int k,y=0, i=0, index_code=0;
    while(i <70418432){
        args->len_k[index_code] = code_len;
        args->k_codes[index_code]=0;
        for(k = 0; k < 14; k++){
            code_len_temp = 0;
            for(i=y*64; i < (y*64)+64; i++){
                //if(index_code==4096)
                //printf(" i=%d, k=%d, value=%hu, shift=%hu\n",i,k,args->ptr_k[i], (args->ptr_k[i] >> k));
                code_len_temp += (args->ptr_k[i] >> k) + 1 + k;
            }
            //printf("for out\n");
    
            if(code_len_temp < args->len_k[index_code]){
                
                args->len_k[index_code] = code_len_temp;
                args->k_codes[index_code] = k;
                
            }
        }
        //printf("post k");
        sem_post(&sem_k);
        if(( args->len_k[index_code]!= h_len[index_code]) || (args->k_codes[index_code] !=h_code[index_code])){
            printf("index=%d,pthread_len=%u, len=%u, pthread_code=%d, code=%d\n", index_code, args->len_k[index_code], h_len[index_code] ,  args->k_codes[index_code], h_code[index_code] );
            exit(0);
        } 
        y++;
        index_code++;
        args->len_k[index_code] = (unsigned int)-1;
    }
    pthread_exit (NULL); */
}

void *compute_stream_arm(void * input){
    struct thread_args_stream *args = (struct thread_args_stream *) input;
    int i=0;
    
    
    //clock_gettime(CLOCK_MONOTONIC, &start4);
    //printf("BITSTREAM_SEM=%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);
    while(i<args->n_blocks){
        //bytes[i]=args->written_bytes[0];
        //bits[i]=args->written_bits[0];
        //printf("bytes=%u, bits=%u\n", args->written_bytes[0],args->written_bits[0]);
        //printf("waiting for zero");
        //sem_wait(&sem_zero);
        //printf("waiting for sec ext");
        
        if (args->stream_zero_codes[i]==1){
            printf("zero block enter\n");
            int k=1;
            while(args->stream_zero_codes[i+k]!=1){
                //sem_wait(&sem_zero);
                if(args->stream_zero_codes[i+k]==1){
                    k++;
                }
                else{
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 5, 0);
                    if(k > 1){
                        if(k < 5){
                            bitStream_store_constant(args->compressed_stream, args->written_bytes,args->written_bits, k - 1, 0);
                        }else{
                            //if(end_of_segment != 0)
                                //bitStream_store_constant(compressed_stream, written_bytes, written_bits, 4, 0);
                            //else
                                bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, k, 0);
                        }
                    }
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                }
            }
            i=i+k;
            if(i>=args->n_blocks)
                break;
        }
        else{
            //sem_wait(&sem_sec_ext);
            //printf("waiting for k");
            //if(i%1024==0)
            //sem_wait(&sem_k);
            

            int chosenMethod = -2;
            unsigned int method_code_size = 1024;

            if(args->stream_len_sec_ext[i] < method_code_size){
                // second extension is best, I go for it
                chosenMethod = -1;
                method_code_size = args->stream_len_sec_ext[i];
            }
            if(args->stream_len_k[i] < method_code_size){
                chosenMethod = args->stream_k_codes[i];
            }
            
            if(chosenMethod >= 0){
                //if(i==136)
                    //printf("ksplit=%d\n", chosenMethod);
                //printf("check=%u\n",args->written_bytes[0]);
                bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, chosenMethod + 1);
                //printf("check\n");
                for(int k = i*64; k < (i*64)+64; k++){
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, (args->ptr_stream[k] >> chosenMethod), 0);
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                }

                for(int k = i*64; k < (i*64)+64; k++){
                    /* if(i==0){
                        printf("bytes=%u, bits=%u, k=%u, ptr=%hhu, args->compressed_stream=%hhu\n",args->written_bytes[0], args->written_bits[0], chosenMethod,args->ptr_stream[k], args->ptr_stream[k-1] );
                    } */
                    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, chosenMethod, args->ptr_stream[k]);
                }
            }
            else if (chosenMethod==-1){
                //if(i==136)
                    //printf("sec\n");
                bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 4, 0);
                bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                
                //if(i==136)

                for(int k = i*32; k < (i*32)+32; k++){
                    
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, args->stream_sec_ext_codes[k], 0);
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                }
                //if(i==136)
                    //printf("check\n");

            }
            else{
                bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 4, 1);

                for(int k = i*64; k < (i*64)+64; k++){
                    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->ptr_stream[k]);
                }
            }
            i++;

        }
        //printf("i=%d\n", i);
    } 
    //printf("bytes=%u, bits=%u\n", args->written_bytes[0],args->written_bits[0]);

    //printf("const_zero=%u, const_less=%u, const_diff_zero=%u, zero=%u, less=%u, diff_zero=%u, for1=%u, for2=%u, big_zero=%u\n", const_zero, const_less, const_diff_zero, zero, less, diff_zero, for1, for2, big_zero);
    /* clock_gettime(CLOCK_MONOTONIC, &end4);
    printf("ARM=%lf\n",(end4.tv_sec-start4.tv_sec)*1e3+(end4.tv_nsec-start4.tv_nsec)*1e-6);
    FILE *fp;
    fp = fopen("tempos_arm.txt","a");
    if(fp == NULL){
        printf("Error writng file!\n");   
        exit(1);             
    }
    fprintf(fp,"%lf\n",(end4.tv_sec-start4.tv_sec)*1e3+(end4.tv_nsec-start4.tv_nsec)*1e-6);
    fclose(fp); */

    pthread_exit (NULL);

}

void *compute_stream_denver(void * input){
    struct thread_args_stream *args = (struct thread_args_stream *) input;
    int i=0;
    
    
    //clock_gettime(CLOCK_MONOTONIC, &start3);
    //printf("BITSTREAM_SEM=%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);
    while(i<args->n_blocks){
        //bytes[i]=args->written_bytes[0];
        //bits[i]=args->written_bits[0];
        //printf("bytes=%u, bits=%u\n", args->written_bytes[0],args->written_bits[0]);
        //printf("waiting for zero");
        //sem_wait(&sem_zero);
        //printf("waiting for sec ext");
        
        if (args->stream_zero_codes[i]==1){
            //printf("zero block enter\n");
            int k=1;
            while(args->stream_zero_codes[i+k]!=1){
                //sem_wait(&sem_zero);
                if(args->stream_zero_codes[i+k]==1){
                    k++;
                }
                else{
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 5, 0);
                    if(k > 1){
                        if(k < 5){
                            bitStream_store_constant(args->compressed_stream, args->written_bytes,args->written_bits, k - 1, 0);
                        }else{
                            //if(end_of_segment != 0)
                                //bitStream_store_constant(compressed_stream, written_bytes, written_bits, 4, 0);
                            //else
                                bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, k, 0);
                        }
                    }
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                }
            }
            i=i+k;
            if(i>=args->n_blocks)
                break;
        }
        else{
            //sem_wait(&sem_sec_ext);
            //printf("waiting for k");
            //if(i%1024==0)
            //sem_wait(&sem_k);
            

            int chosenMethod = -2;
            unsigned int method_code_size = 1024;

            if(args->stream_len_sec_ext[i] < method_code_size){
                // second extension is best, I go for it
                chosenMethod = -1;
                method_code_size = args->stream_len_sec_ext[i];
            }
            if(args->stream_len_k[i] < method_code_size){
                chosenMethod = args->stream_k_codes[i];
            }
            
            if(chosenMethod >= 0){
                //if(i==136)
                    //printf("ksplit=%d\n", chosenMethod);
                //printf("check=%u\n",args->written_bytes[0]);
                bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, chosenMethod + 1);
                //printf("check\n");
                for(int k = i*64; k < (i*64)+64; k++){
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, (args->ptr_stream[k] >> chosenMethod), 0);
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                }

                for(int k = i*64; k < (i*64)+64; k++){
                    /* if(i==0){
                        printf("bytes=%u, bits=%u, k=%u, ptr=%hhu, args->compressed_stream=%hhu\n",args->written_bytes[0], args->written_bits[0], chosenMethod,args->ptr_stream[k], args->ptr_stream[k-1] );
                    } */
                    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, chosenMethod, args->ptr_stream[k]);
                }
            }
            else if (chosenMethod==-1){
                //if(i==136)
                    //printf("sec\n");
                bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 4, 0);
                bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                
                //if(i==136)

                for(int k = i*32; k < (i*32)+32; k++){
                    
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, args->stream_sec_ext_codes[k], 0);
                    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
                }
                //if(i==136)
                    //printf("check\n");

            }
            else{
                bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 4, 1);

                for(int k = i*64; k < (i*64)+64; k++){
                    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->ptr_stream[k]);
                }
            }
            i++;

        }
        //printf("i=%d\n", i);
    } 
    //printf("bytes=%u, bits=%u\n", args->written_bytes[0],args->written_bits[0]);

    //printf("const_zero=%u, const_less=%u, const_diff_zero=%u, zero=%u, less=%u, diff_zero=%u, for1=%u, for2=%u, big_zero=%u\n", const_zero, const_less, const_diff_zero, zero, less, diff_zero, for1, for2, big_zero);
    /* clock_gettime(CLOCK_MONOTONIC, &end3);
    printf("DENVER=%lf\n",(end3.tv_sec-start3.tv_sec)*1e3+(end3.tv_nsec-start3.tv_nsec)*1e-6);
    FILE *fp;
    fp = fopen("tempos_denver.txt","a");
    if(fp == NULL){
        printf("Error writng file!\n");   
        exit(1);             
    }
    fprintf(fp,"%lf\n",(end3.tv_sec-start3.tv_sec)*1e3+(end3.tv_nsec-start3.tv_nsec)*1e-6);
    fclose(fp); */

    pthread_exit (NULL);

}

void *compute_stream0(void * input){
    struct thread_args_stream_conc *args = (struct thread_args_stream_conc *) input;
    //clock_gettime(CLOCK_MONOTONIC, &start2);

    /* IMAGE METADATA */
    // User defined data
    /* bitStream_store_constant(compressed_stream, written_bytes, written_bits, 8, 0);
    // x, y, z dimensions
    bitStream_store(compressed_stream, written_bytes, written_bits, 16, args->input_params.x_size);
    bitStream_store(compressed_stream, written_bytes, written_bits, 16, args->input_params.y_size);
    bitStream_store(compressed_stream, written_bytes, written_bits, 16, args->input_params.z_size);
    // Sample type
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
    // dynamic range
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, 16);
    // Encoding Sample Order and interleaving
    if(encoder_params.out_interleaving == BSQ){
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 16, 0);
    }
    else{
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
        bitStream_store(compressed_stream, written_bytes, written_bits, 16, encoder_params.out_interleaving_depth);
    }
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
    // Out word size
    bitStream_store(compressed_stream, written_bytes, written_bits, 3, 1);
    // Encoder type
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 10, 0);

    /* PREDICTOR METADATA */
    // reserved
    /*bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
    // prediction bands
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, predictor_params.user_input_pred_bands);
    // prediction mode
    if(predictor_params.full != 0)
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    else
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // local sum
    if(predictor_params.neighbour_sum != 0)
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    else
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // Register size
    bitStream_store(compressed_stream, written_bytes, written_bits, 6, 32);
    // Weight resolution
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, 0);
    // weight update scaling exponent change interval
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, ((unsigned int)log2((float)2048)) - 4);
    // weight update scaling exponent initial parameter
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, 0);
    // weight update scaling exponent final parameter
    bitStream_store(compressed_stream, written_bytes, written_bits, 4, 0);
    // reserved
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // weight initialization method and weight initialization table flag
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
    // weight initialization resolution
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 5, 0);
    // Weight initialization table

    /* ENTROPY CODER METADATA */
    // reserved
    /*bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
    // block size
    bitStream_store(compressed_stream, written_bytes, written_bits, 2, 0x3);

        // Restricted code
    bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
        // Reference Sample Interval
    bitStream_store(compressed_stream, written_bytes, written_bits, 12, 1); */



    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 8, 0);
    // x, y, z dimensions
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->input_params.x_size);
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->input_params.y_size);
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->input_params.z_size);
    // Sample type
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0);
    // dynamic range
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 16);
    // Encoding Sample Order and interleaving

    if(args->encoder_params.out_interleaving == BSQ){
        bitStream_store_constant(args->compressed_stream, args->written_bytes,args->written_bits, 1, 1);
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 16, 0);
    }
    else{
        bitStream_store_constant(args->compressed_stream, args->written_bytes,args-> written_bits, 1, 0);
        bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 16, args->encoder_params.out_interleaving_depth);
    }
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0);
    // Out word size
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 3, 1);
    // Encoder type
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 10, 0);

    /* PREDICTOR METADATA */
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0);
    // prediction bands
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, args->predictor_params.user_input_pred_bands);
    // prediction mode
    if(args->predictor_params.full != 0)
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    else
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // local sum
    if(args->predictor_params.neighbour_sum != 0)
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    else
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 1);

    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // Register size
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 6, 32);
    // Weight resolution
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 0);
    // weight update scaling exponent change interval
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, ((unsigned int)log2((float)2048)) - 4);
    // weight update scaling exponent initial parameter
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 0);
    // weight update scaling exponent final parameter
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 4, 0);
    // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
    // weight initialization method and weight initialization table flag

    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0);
    // weight initialization resolution

    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 5, 0);
    // Weight initialization table


    /* ENTROPY CODER METADATA */

        // reserved
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
        // block size
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 2, 0x3);

        // Restricted code
    bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, 1, 0);
        // Reference Sample Interval
    bitStream_store(args->compressed_stream, args->written_bytes, args->written_bits, 12, 1);
    //clock_gettime(CLOCK_MONOTONIC, &end2);
    //printf("BITSTREAM_HEADER=%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);
    

    //clock_gettime(CLOCK_MONOTONIC, &end2);
    //printf("BITSTREAM_SEM=%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes1[0], args->written_bits1[0]);

    int i=0;
    if(args->written_bits[0]==0){
        for(i=0; i<=args->written_bytes1[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream1[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes1[0];i++){
            //if(i==args->written_bytes1[0])
            //printf("aux=%u\n", args->written_bytes1[0]);
            args->compressed_stream[args->written_bytes[0]]|=(args->stream1[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream1[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits1[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream1[i]>>args->written_bits[0]);
            args->written_bytes[0]--;

        }
        /*else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream1[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream1[i]<<(args->written_bits1[0]-(args->written_bits1[0]+args->written_bits[0])%64));

        } */
    }
    args->written_bits[0]=(args->written_bits[0]+args->written_bits1[0])%64;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes2[0], args->written_bits2[0]);


    if(args->written_bits[0]==0){
        for(i=0; i<args->written_bytes2[0+1];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream2[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes2[0];i++){

            args->compressed_stream[args->written_bytes[0]]|=(args->stream2[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream2[i]<<(64-args->written_bits[0]));


        }
         if(args->written_bits[0]+args->written_bits2[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream2[i]>>args->written_bits[0]);
            args->written_bytes[0]--;

        }
        /*else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream2[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream2[i]<<(args->written_bits2[0]-(args->written_bits2[0]+args->written_bits[0])%64));

        } */
    }

    args->written_bits[0]=(args->written_bits[0]+args->written_bits2[0])%64;
    //args->written_bytes[0]--;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes3[0], args->written_bits3[0]);



    if(args->written_bits[0]==0){
        for(i=0; i<args->written_bytes3[0+1];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream3[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes3[0];i++){

            args->compressed_stream[args->written_bytes[0]]|=(args->stream3[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream3[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits3[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream3[i]>>args->written_bits[0]);
            args->written_bytes[0]--;
        }
        /* else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream3[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream3[i]<<(args->written_bits3[0]-(args->written_bits3[0]+args->written_bits[0])%64));

        } */
    }

    args->written_bits[0]=(args->written_bits[0]+args->written_bits3[0])%64;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes4[0], args->written_bits4[0]);



    if(args->written_bits[0]==0){
        for(i=0; i<args->written_bytes4[0+1];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream4[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes4[0];i++){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream4[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream4[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits4[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream4[i]>>args->written_bits[0]);
            args->written_bytes[0]--;
        }
        /* else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream4[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream4[i]<<(args->written_bits4[0]-(args->written_bits4[0]+args->written_bits[0])%64));

        } */
    }

    args->written_bits[0]=(args->written_bits[0]+args->written_bits4[0])%64;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes5[0], args->written_bits5[0]);


    if(args->written_bits[0]==0){
        for(i=0; i<args->written_bytes5[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream5[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes5[0];i++){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream5[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream5[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits5[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream5[i]>>args->written_bits[0]);
            args->written_bytes[0]--;
        }
        /* else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream5[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream5[i]<<(args->written_bits5[0]-(args->written_bits5[0]+args->written_bits[0])%64));

        } */
    }

    args->written_bits[0]=(args->written_bits[0]+args->written_bits5[0])%64;
    //printf("byte=%u, bit=%u, stream_byte=%u, stream_bit=%u\n", args->written_bytes[0],args->written_bits[0],args->written_bytes6[0], args->written_bits6[0]);



    if(args->written_bits[0]==0){
        for(i=0; i<=args->written_bytes6[0];i++){
            args->compressed_stream[args->written_bytes[0]]=args->stream6[i];
            args->written_bytes[0]++;
        }
    }
    else{
        for(i=0; i<=args->written_bytes6[0];i++){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream6[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream6[i]<<(64-args->written_bits[0]));
        }
        if(args->written_bits[0]+args->written_bits6[0] <64){
            args->compressed_stream[args->written_bytes[0]]|=(args->stream6[i]>>args->written_bits[0]);
            args->written_bytes[0]--;
        }
/*         else{
            args->compressed_stream[args->written_bytes[0]]|=(args->stream5[i]>>args->written_bits[0]);
            args->written_bytes[0]++;
            args->compressed_stream[args->written_bytes[0]]=(args->stream5[i]<<(args->written_bits5[0]-(args->written_bits5[0]+args->written_bits[0])%64));

        } */
    }

    args->written_bits[0]=(args->written_bits[0]+args->written_bits6[0])%64;
    //printf("byte=%u, bit=%u\n", args->written_bytes[0],args->written_bits[0]);


    //printf("bytes=%u, bits=%u\n", args->written_bytes[0],args->written_bits[0]);

    //printf("const_zero=%u, const_less=%u, const_diff_zero=%u, zero=%u, less=%u, diff_zero=%u, for1=%u, for2=%u, big_zero=%u\n", const_zero, const_less, const_diff_zero, zero, less, diff_zero, for1, for2, big_zero);
    unsigned char num_padding_bits=args->written_bits[0]%8;
    //printf("num_padding_bits=%hhu\n", num_padding_bits);
    if(num_padding_bits < 8 && num_padding_bits > 0){
        bitStream_store_constant(args->compressed_stream, args->written_bytes, args->written_bits, num_padding_bits, 0);
    }
    //clock_gettime(CLOCK_MONOTONIC, &end2);
    //printf("BITSTREAM=%lf\n",(end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_nsec-start2.tv_nsec)*1e-6);

    pthread_exit (NULL);

}



/// This procedure computes the codes for all the different k-split, second-extension and
/// no compression options and encodes the block according to the code yielding
/// the highest compression factor.
extern "C" void compute_block_code(input_feature_t input_params, encoder_config_t encoder_params, 
    unsigned short int * block_samples, unsigned long * compressed_stream, unsigned int * written_bytes, unsigned int * written_bits){
    // I encode the chosen method as the value of k for k-split;
    // second-extension is -1 and no compression -2
    int chosenMethod = -2;
    unsigned int method_code_size = input_params.dyn_range*encoder_params.block_size;
    unsigned int second_extension_values[32];
    int k_split = 0;
    int i = 0;

    args_second->block_samples=block_samples;
    args_second->second_extension_values=second_extension_values;

    args_ksplit->block_samples=block_samples;
    args_ksplit->k_split=&k_split;

    


    // First of all I compute which method is the one yielding smaller compression; note that
    // the second extension compression method has the block ID 1 bit longer
    //clock_gettime(CLOCK_MONOTONIC, &start);
    //double StartTime=omp_get_wtime();
    pthread_create(&threads[0], &(threads_attr[0]), compute_second_extension, args_second);
    pthread_create(&threads[2], &(threads_attr[2]), compute_ksplit, args_ksplit);
    pthread_join(threads[0], NULL);
    pthread_join(threads[2], NULL);
    //printf("time= %lf\n", omp_get_wtime()-StartTime);

    

    //temp_size = compute_second_extension(encoder_params, block_samples, second_extension_values) + 1;
    //clock_gettime(CLOCK_MONOTONIC, &end);
    //printf("%lf\n",(end.tv_sec-start.tv_sec)*1e3+(end.tv_nsec-start.tv_nsec)*1e-6);
    //exit(0);
    if(args_second->return_arg < method_code_size){
        // second extension is best, I go for it
        chosenMethod = -1;
        method_code_size = args_second->return_arg;
    }
    // Now we have to analyze the k-split
    if(input_params.dyn_range > 2 || encoder_params.restricted == 0){
        //if(compute_ksplit(input_params, encoder_params, block_samples, &k_split, *written_bytes > 0x2000 && *written_bytes < 0x3e10) < method_code_size){
        if(args_ksplit->return_arg < method_code_size){
            // second extension is best, I go for it
            chosenMethod = k_split;
        }
    }
    // Done, best method chosen. Let's perform the compression, adding the block codes
    // to the bitstream.
    if(args_ksplit->return_arg==3)
            printf("ksplit=%d\n", chosenMethod);
    if(chosenMethod == -2){
        // no compression
        //First of all I have to save the ID of the block code option
        if(input_params.dyn_range <= 4 && encoder_params.restricted != 0){
            if(input_params.dyn_range < 3)
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
            else
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 1);
        }else{
            if(input_params.dyn_range <= 8){
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 3, 1);
            }else if(input_params.dyn_range <= 16){
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 4, 1);
            }else{
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 5, 1);
            }
        }
        // and now the codes for the samples
        for(i = 0; i < encoder_params.block_size; i++){
            bitStream_store(compressed_stream, written_bytes, written_bits, input_params.dyn_range, block_samples[i]);
        }
    }else if(chosenMethod == -1){
        
        // second extension option
        //First of all I have to save the ID of the block code option
        if(input_params.dyn_range <= 4 && encoder_params.restricted != 0){
            if(input_params.dyn_range < 3)
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 0);
            else
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 2, 0);
        }else{
            if(input_params.dyn_range <= 8){
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 3, 0);
            }else if(input_params.dyn_range <= 16){
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 4, 0);
            }else{
                bitStream_store_constant(compressed_stream, written_bytes, written_bits, 5, 0);
            }
        }
        bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
        //if(aux2==270)
        //printf("inside orig_bytes=%u, orig_bits=%u, thread_bytes=%u, thread_bits=%u, k=%d\n",written_bytes[0],written_bits[0], bytes[aux2], bits[aux2], aux2);
        // and now the codes for the samples
        for(i = 0; i < encoder_params.block_size/2; i++){
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, second_extension_values[i], 0);
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
        }
       
    }else{
        
        // k-split
        //First of all I have to save the ID of the block code option
        if(input_params.dyn_range <= 4 && encoder_params.restricted != 0){
            bitStream_store(compressed_stream, written_bytes, written_bits, 2, chosenMethod + 1);
        }else{
            if(input_params.dyn_range <= 8){
                bitStream_store(compressed_stream, written_bytes, written_bits, 3, chosenMethod + 1);
            }else if(input_params.dyn_range <= 16){
                bitStream_store(compressed_stream, written_bytes, written_bits, 4, chosenMethod + 1);
            }else{
                bitStream_store(compressed_stream, written_bytes, written_bits, 5, chosenMethod + 1);
            }
        }
        // and now the codes for the samples
        for(i = 0; i < encoder_params.block_size; i++){
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, (block_samples[i] >> chosenMethod), 0);
            bitStream_store_constant(compressed_stream, written_bytes, written_bits, 1, 1);
        }
        for(i = 0; i < encoder_params.block_size; i++){
            bitStream_store(compressed_stream, written_bytes, written_bits, chosenMethod, block_samples[i]);
        }
    }
}

extern "C" int create_block(input_feature_t input_params, encoder_config_t encoder_params, unsigned short int * block_samples, int all_zero,
    int * num_zero_blocks, int * segment_idx, int reference_samples,
    unsigned long * compressed_stream, unsigned int * written_bytes, unsigned int * written_bits){
    // I have finished reading the block: we now need to pass it to the compressor, unless
    // it is an all zero block
    if(all_zero == 0){
        
        // Before encoding this block I have to check if there are zero-block that
        // have already been counted and that need to be encoded
        if(*num_zero_blocks > 0){
            printf("check\n");

            zero_block_code(input_params, encoder_params, *num_zero_blocks, compressed_stream, written_bytes, written_bits, 0);
            *num_zero_blocks = 0;
        }
        //printf("check\n");
        compute_block_code(input_params, encoder_params, block_samples, compressed_stream, written_bytes, written_bits);
    }else{
        printf("check\n");
        *num_zero_blocks = *num_zero_blocks + 1;
    }
    *segment_idx = *segment_idx + 1;
    if(*segment_idx == SEGMENT_SIZE || reference_samples == encoder_params.ref_interval){
        // I arrived at the end of a segment: if there are un-encoded
        // zero_blocks I encode them
        

        if(*num_zero_blocks > 0){
            printf("check\n");
            zero_block_code(input_params, encoder_params, *num_zero_blocks, compressed_stream, written_bytes, written_bits, 1);
            *num_zero_blocks = 0;
        }
        *segment_idx = 0;
    }
//#ifndef NDEBUG
    if(*written_bytes > input_params.dyn_range*input_params.x_size*input_params.y_size*input_params.z_size){
        fprintf(stderr, "Error in create_block, writing outside the compressed_stream boundaries: it means that the compressed image is greater than the original\n");
        return -1;
    }
//#endif
    return 0;
}


///Given the characteristics of the input stream, the parameters describing the desired behavior
///of the encoder and the list of residuals to be encoded (note that each residual is treated as
///an integer) it returs the size in bytes of the stream containing the compressed residuals (saved into compressed_stream)
///After usage, the caller has to deallocate the memory area pointed by compressed_stream and allocated by this function.
///@param input_params describe the image whose residuals are contained in the input file
///@param encoder_params set of options determining the behavior of the encoder
///@param residuals array containing the information to be compressed
///@param compressed_stream pointer to the array which, at the end of this function, will contain the compressed information
///the array is allocated inside this function
///@return a negative number if an error occurred
extern "C" int encode_block(input_feature_t input_params, predictor_config_t predictor_params, encoder_config_t encoder_params, unsigned short int * residuals,
    unsigned long * compressed_stream, unsigned int * written_bytes, unsigned int * written_bits){
    // First of all I have to pick-up the J elements composing a block and
    // then pass them to the compressor; if a block is composed of
    // all zeros, then I read the remaining blocks in the segment, up
    // to the first block containing a non-zero sample.
    // Let's remember that the elements are saved in residuals so that
    // element(x, y, z) = residuals[x + y*x_size + z*x_size*y_size], i.e.
    // they are saved in BSQ order
    int read_samples = 0;
    int all_zero = 1;
    int num_zero_blocks = 0;
    //int reference_samples = 0;

    if(mlockall(MCL_CURRENT|MCL_FUTURE) == -1)
        printf("mlockall\n");

        
    unsigned short int * block_samples = NULL;
    unsigned short int * ptr_zero_block=NULL;
    unsigned short int * ptr_sec_ext=NULL;
    unsigned short int * ptr_k=NULL;
    unsigned short int * ptr_stream=NULL;
    


    unsigned int * len_sec_ext=NULL;
    unsigned int * len_k=NULL;
    /* unsigned short int * ptr_k1=NULL;
    unsigned short int * ptr_k2=NULL;
    unsigned short int * ptr_k3=NULL;
    unsigned short int * ptr_k=NULL;
    unsigned short int * ptr_k4=NULL;
    unsigned short int * ptr_k5=NULL;
    unsigned short int * ptr_k6=NULL;
    unsigned short int * ptr_k7=NULL;
    unsigned short int * ptr_k8=NULL;
    unsigned short int * ptr_k9=NULL;
    unsigned short int * ptr_k10=NULL;
    unsigned short int * ptr_k11=NULL;
    unsigned short int * ptr_k12=NULL;
    unsigned short int * ptr_k13=NULL;
 */
    

    
    unsigned char * zero_codes=NULL;
    unsigned int * sec_ext_codes=NULL;
    int * k_codes=NULL;

    unsigned char * stream_zero_codes1=NULL;
    unsigned int * stream_len_sec_ext1=NULL;
    unsigned int * stream_len_k1=NULL;
    int * stream_k_codes1=NULL;
    unsigned int * stream_sec_ext_codes1=NULL;
    unsigned long * stream1=NULL;
    unsigned int * written_bytes1=NULL;
    unsigned int * written_bits1=NULL;


    unsigned char * stream_zero_codes2=NULL;
    unsigned int * stream_len_sec_ext2=NULL;
    unsigned int * stream_len_k2=NULL;
    int * stream_k_codes2=NULL;
    unsigned int * stream_sec_ext_codes2=NULL;
    unsigned long * stream2=NULL;
    unsigned int * written_bytes2=NULL;
    unsigned int * written_bits2=NULL;

    unsigned char * stream_zero_codes3=NULL;
    unsigned int * stream_len_sec_ext3=NULL;
    unsigned int * stream_len_k3=NULL;
    int * stream_k_codes3=NULL;
    unsigned int * stream_sec_ext_codes3=NULL;
    unsigned long * stream3=NULL;
    unsigned int * written_bytes3=NULL;
    unsigned int * written_bits3=NULL;

    unsigned char * stream_zero_codes4=NULL;
    unsigned int * stream_len_sec_ext4=NULL;
    unsigned int * stream_len_k4=NULL;
    int * stream_k_codes4=NULL;
    unsigned int * stream_sec_ext_codes4=NULL;
    unsigned long * stream4=NULL;
    unsigned int * written_bytes4=NULL;
    unsigned int * written_bits4=NULL;

    unsigned char * stream_zero_codes5=NULL;
    unsigned int * stream_len_sec_ext5=NULL;
    unsigned int * stream_len_k5=NULL;
    int * stream_k_codes5=NULL;
    unsigned int * stream_sec_ext_codes5=NULL;
    unsigned long * stream5=NULL;
    unsigned int * written_bytes5=NULL;
    unsigned int * written_bits5=NULL;

    unsigned char * stream_zero_codes6=NULL;
    unsigned int * stream_len_sec_ext6=NULL;
    unsigned int * stream_len_k6=NULL;
    int * stream_k_codes6=NULL;
    unsigned int * stream_sec_ext_codes6=NULL;
    unsigned long * stream6=NULL;
    unsigned int * written_bytes6=NULL;
    unsigned int * written_bits6=NULL;

    //cudaError_t err=cudaSuccess;
    //size_t size_len=sizeof(unsigned int)*(1100288);
    //size_t size_codes=(sizeof(int)*(1100288));

    /* err=cudaHostAlloc((void **)&k_codes, size_codes, cudaHostAllocDefault );
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate host mpr(error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err=cudaHostAlloc((void **)&len_k, size_len, cudaHostAllocDefault );
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate host mpr(error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    } */

    




    ptr_zero_block=(unsigned short int *)malloc(sizeof(unsigned short int));
    if(ptr_zero_block == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    ptr_sec_ext=(unsigned short int *)malloc(sizeof(unsigned short int));
    if(ptr_sec_ext == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    ptr_k=(unsigned short int *)malloc(sizeof(unsigned short int));
    if(ptr_k == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    ptr_stream=(unsigned short int *)malloc(sizeof(unsigned short int));
    if(ptr_stream == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
 

    
    //printf("value=%u\n", compressed_stream[0]);
    len_sec_ext=(unsigned int *)malloc(sizeof(unsigned int)*((input_params.x_size*input_params.y_size*input_params.z_size)/encoder_params.block_size));
    if(len_sec_ext == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    len_k=(unsigned int *)malloc(sizeof(unsigned int)*(((input_params.x_size*input_params.y_size*input_params.z_size)/encoder_params.block_size)+512));
    if(len_k == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }



    zero_codes=(unsigned char *)malloc(sizeof(unsigned char)*((input_params.x_size*input_params.y_size*input_params.z_size)/encoder_params.block_size));
    if(zero_codes == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    sec_ext_codes=(unsigned int *)malloc(sizeof(unsigned int)*((input_params.x_size*input_params.y_size*input_params.z_size)/2));
    if(sec_ext_codes == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    k_codes=(int *)malloc(sizeof(int)*(((input_params.x_size*input_params.y_size*input_params.z_size)/encoder_params.block_size)+512));
    if(k_codes == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    //==============================================================================================================================
    //stream 1
        stream_zero_codes1=(unsigned char *)malloc(sizeof(unsigned char));
        if(stream_zero_codes1 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream_len_sec_ext1=(unsigned int *)malloc(sizeof(unsigned int));
        if(stream_len_sec_ext1 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream_len_k1=(unsigned int *)malloc(sizeof(unsigned int));
        if(stream_len_k1 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream_k_codes1=( int *)malloc(sizeof( int));
        if(stream_k_codes1 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream_sec_ext_codes1=(unsigned int *)malloc(sizeof(unsigned int));
        if(stream_sec_ext_codes1 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream1=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
        if(stream1 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }
        memset(stream1, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

        written_bytes1=(unsigned int *)malloc(sizeof(unsigned int));
        if(written_bytes1 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        written_bits1=(unsigned int *)malloc(sizeof(unsigned int));
        if(written_bits1 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        written_bytes1[0]=0;
        written_bits1[0]=0;


    //==============================================================================================================================
    //stream 2
        stream_zero_codes2=(unsigned char *)malloc(sizeof(unsigned char));
        if(stream_zero_codes2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream_len_sec_ext2=(unsigned int *)malloc(sizeof(unsigned int));
        if(stream_len_sec_ext2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream_len_k2=(unsigned int *)malloc(sizeof(unsigned int));
        if(stream_len_k2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream_k_codes2=( int *)malloc(sizeof( int));
        if(stream_k_codes2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream_sec_ext_codes2=(unsigned int *)malloc(sizeof(unsigned int));
        if(stream_sec_ext_codes2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        stream2=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
        if(stream2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }
        memset(stream2, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

        written_bytes2=(unsigned int *)malloc(sizeof(unsigned int));
        if(written_bytes2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        written_bits2=(unsigned int *)malloc(sizeof(unsigned int));
        if(written_bits2 == NULL){
            printf("Failed to allocate host mpr\n" );
            exit(EXIT_FAILURE);
        }

        written_bytes2[0]=0;
        written_bits2[0]=0;

    //==============================================================================================================================
    //stream 3
    stream_zero_codes3=(unsigned char *)malloc(sizeof(unsigned char));
    if(stream_zero_codes3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_len_sec_ext3=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_len_sec_ext3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_len_k3=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_len_k3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_k_codes3=( int *)malloc(sizeof( int));
    if(stream_k_codes3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_sec_ext_codes3=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_sec_ext_codes3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream3=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream3, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    written_bytes3=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits3=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits3 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes3[0]=0;
    written_bits3[0]=0;

    //==============================================================================================================================
    //stream 4
    stream_zero_codes4=(unsigned char *)malloc(sizeof(unsigned char));
    if(stream_zero_codes4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_len_sec_ext4=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_len_sec_ext4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_len_k4=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_len_k4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_k_codes4=( int *)malloc(sizeof( int));
    if(stream_k_codes4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_sec_ext_codes4=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_sec_ext_codes4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream4=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream4, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    written_bytes4=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits4=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits4 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes4[0]=0;
    written_bits4[0]=0;

    //==============================================================================================================================
    //stream 5
    stream_zero_codes5=(unsigned char *)malloc(sizeof(unsigned char));
    if(stream_zero_codes5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_len_sec_ext5=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_len_sec_ext5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_len_k5=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_len_k5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_k_codes5=( int *)malloc(sizeof( int));
    if(stream_k_codes5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_sec_ext_codes5=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_sec_ext_codes5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream5=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream5, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    written_bytes5=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits5=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits5 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes5[0]=0;
    written_bits5[0]=0;

    //==============================================================================================================================
    //stream 6
    stream_zero_codes6=(unsigned char *)malloc(sizeof(unsigned char));
    if(stream_zero_codes6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_len_sec_ext6=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_len_sec_ext6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_len_k6=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_len_k6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_k_codes6=( int *)malloc(sizeof( int));
    if(stream_k_codes6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream_sec_ext_codes6=(unsigned int *)malloc(sizeof(unsigned int));
    if(stream_sec_ext_codes6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    stream6=(unsigned long *)malloc((((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(stream6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }
    memset(stream6, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    written_bytes6=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bytes6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bits6=(unsigned int *)malloc(sizeof(unsigned int));
    if(written_bits6 == NULL){
        printf("Failed to allocate host mpr\n" );
        exit(EXIT_FAILURE);
    }

    written_bytes6[0]=0;
    written_bits6[0]=0;





  
    //evita que a memoria seja usada para swap
    unsigned int arm=((input_params.x_size*input_params.y_size*input_params.z_size)/64)/10;
    unsigned int denver=(((input_params.x_size*input_params.y_size*input_params.z_size)/64)*30)/100;
    unsigned int rem_arm=(((input_params.x_size*input_params.y_size*input_params.z_size)/64)%10)*4;
    unsigned int rem_denver=(((((input_params.x_size*input_params.y_size*input_params.z_size)/64)*30)%100)/10)*2;
    //arm=arm+((rem_arm+rem_denver)/10)/4;
    //denver++;
    printf("arm=%um denver=%u, rem arm=%u, rem denver=%u\n", arm, denver, rem_arm, rem_denver);
    

    args_second = (struct thread_args_second *)malloc(sizeof(struct thread_args_second));
    args_ksplit = (struct thread_args_split *)malloc(sizeof(struct thread_args_split));
    args_zero = (struct thread_args_zero *)malloc(sizeof(struct thread_args_zero));
    args_sec_ext = (struct thread_args_sec_ext *)malloc(sizeof(struct thread_args_sec_ext));
    args_k = (struct thread_args_k *)malloc(sizeof(struct thread_args_k));
    args_stream0 = (struct thread_args_stream_conc *)malloc(sizeof(struct thread_args_stream_conc));
    args_stream1 = (struct thread_args_stream *)malloc(sizeof(struct thread_args_stream));
    args_stream2 = (struct thread_args_stream *)malloc(sizeof(struct thread_args_stream));
    args_stream3 = (struct thread_args_stream *)malloc(sizeof(struct thread_args_stream));
    args_stream4 = (struct thread_args_stream *)malloc(sizeof(struct thread_args_stream));
    args_stream5 = (struct thread_args_stream *)malloc(sizeof(struct thread_args_stream));
    args_stream6 = (struct thread_args_stream *)malloc(sizeof(struct thread_args_stream));

    args_second->encoder_params=encoder_params;
    args_ksplit->encoder_params=encoder_params;
    args_ksplit->input_params=input_params;

    args_zero->ptr_zero_block=&residuals[0];
    args_zero->zero_codes=&zero_codes[0];
    args_zero->size=input_params.x_size*input_params.y_size*input_params.z_size;

    args_sec_ext->ptr_sec_ext=&residuals[0];
    args_sec_ext->len_sec_ext=&len_sec_ext[0];
    args_sec_ext->sec_ext_codes=&sec_ext_codes[0];
    args_sec_ext->size=input_params.x_size*input_params.y_size*input_params.z_size;

    args_k->ptr_k=&residuals[0];
    args_k->len_k=&len_k[0];
    args_k->k_codes=&k_codes[0];
    args_k->size=input_params.x_size*input_params.y_size*input_params.z_size;


    args_stream0->compressed_stream=&compressed_stream[0];
    args_stream0->written_bytes=&written_bytes[0];
    args_stream0->written_bits=&written_bits[0];
    args_stream0->stream1=&stream1[0];
    args_stream0->written_bytes1=&written_bytes1[0];
    args_stream0->written_bits1=&written_bits1[0];
    args_stream0->stream2=&stream2[0];
    args_stream0->written_bytes2=&written_bytes2[0];
    args_stream0->written_bits2=&written_bits2[0];
    args_stream0->stream3=&stream3[0];
    args_stream0->written_bytes3=&written_bytes3[0];
    args_stream0->written_bits3=&written_bits3[0];
    args_stream0->stream4=&stream4[0];
    args_stream0->written_bytes4=&written_bytes4[0];
    args_stream0->written_bits4=&written_bits4[0];
    args_stream0->stream5=&stream5[0];
    args_stream0->written_bytes5=&written_bytes5[0];
    args_stream0->written_bits5=&written_bits5[0];
    args_stream0->stream6=&stream6[0];
    args_stream0->written_bytes6=&written_bytes6[0];
    args_stream0->written_bits6=&written_bits6[0];
    args_stream0->input_params=input_params;
    args_stream0->predictor_params=predictor_params;
    args_stream0->encoder_params=encoder_params;


    args_stream1->ptr_stream=&residuals[0];
    args_stream1->stream_zero_codes=&zero_codes[0];
    args_stream1->stream_len_sec_ext=&len_sec_ext[0];
    args_stream1->stream_len_k=&len_k[0];
    args_stream1->stream_k_codes=&k_codes[0];
    args_stream1->stream_sec_ext_codes=&sec_ext_codes[0];
    args_stream1->compressed_stream=&stream1[0];
    args_stream1->written_bytes=&written_bytes1[0];
    args_stream1->written_bits=&written_bits1[0];
    args_stream1->n_blocks=denver;



    args_stream2->ptr_stream=&residuals[denver*64];
    args_stream2->stream_zero_codes=&zero_codes[denver];
    args_stream2->stream_len_sec_ext=&len_sec_ext[denver];
    args_stream2->stream_len_k=&len_k[denver];
    args_stream2->stream_k_codes=&k_codes[denver];
    args_stream2->stream_sec_ext_codes=&sec_ext_codes[(denver*64)/2];
    args_stream2->compressed_stream=&stream2[0];
    args_stream2->written_bytes=&written_bytes2[0];
    args_stream2->written_bits=&written_bits2[0];
    args_stream2->n_blocks=denver;


    args_stream3->ptr_stream=&residuals[(denver*2)*64];
    args_stream3->stream_zero_codes=&zero_codes[(denver*2)];
    args_stream3->stream_len_sec_ext=&len_sec_ext[(denver*2)];
    args_stream3->stream_len_k=&len_k[(denver*2)];
    args_stream3->stream_k_codes=&k_codes[(denver*2)];
    args_stream3->stream_sec_ext_codes=&sec_ext_codes[((denver*2)*64)/2];
    args_stream3->compressed_stream=&stream3[0];
    args_stream3->written_bytes=&written_bytes3[0];
    args_stream3->written_bits=&written_bits3[0];
    args_stream3->n_blocks=arm;



    args_stream4->ptr_stream=&residuals[((denver*2)+arm)*64];
    args_stream4->stream_zero_codes=&zero_codes[((denver*2)+arm)];
    args_stream4->stream_len_sec_ext=&len_sec_ext[((denver*2)+arm)];
    args_stream4->stream_len_k=&len_k[((denver*2)+arm)];
    args_stream4->stream_k_codes=&k_codes[((denver*2)+arm)];
    args_stream4->stream_sec_ext_codes=&sec_ext_codes[(((denver*2)+arm)*64)/2];
    args_stream4->compressed_stream=&stream4[0];
    args_stream4->written_bytes=&written_bytes4[0];
    args_stream4->written_bits=&written_bits4[0];
    args_stream4->n_blocks=arm;




    args_stream5->ptr_stream=&residuals[((denver*2)+(arm*2))*64];
    args_stream5->stream_zero_codes=&zero_codes[((denver*2)+(arm*2))];
    args_stream5->stream_len_sec_ext=&len_sec_ext[((denver*2)+(arm*2))];
    args_stream5->stream_len_k=&len_k[((denver*2)+(arm*2))];
    args_stream5->stream_k_codes=&k_codes[((denver*2)+(arm*2))];
    args_stream5->stream_sec_ext_codes=&sec_ext_codes[(((denver*2)+(arm*2))*64)/2];
    args_stream5->compressed_stream=&stream5[0];
    args_stream5->written_bytes=&written_bytes5[0];
    args_stream5->written_bits=&written_bits5[0];
    args_stream5->n_blocks=arm;




    args_stream6->ptr_stream=&residuals[((denver*2)+(arm*3))*64];
    args_stream6->stream_zero_codes=&zero_codes[((denver*2)+(arm*3))];
    args_stream6->stream_len_sec_ext=&len_sec_ext[((denver*2)+(arm*3))];
    args_stream6->stream_len_k=&len_k[((denver*2)+(arm*3))];
    args_stream6->stream_k_codes=&k_codes[((denver*2)+(arm*3))];
    args_stream6->stream_sec_ext_codes=&sec_ext_codes[(((denver*2)+(arm*3))*64)/2];
    args_stream6->compressed_stream=&stream6[0];
    args_stream6->written_bytes=&written_bytes6[0];
    args_stream6->written_bits=&written_bits6[0];
    args_stream6->n_blocks=arm;


    



    sem_init(&sem_zero,0,0);
    sem_init(&sem_sec_ext,0,0);
    sem_init(&sem_k,0,0);

    CPU_ZERO(&cpuset_zero);
    CPU_SET(0, &cpuset_zero);

    CPU_ZERO(&cpuset_sec_ext);
    //CPU_SET(3, &cpuset_sec_ext);
    CPU_SET(1, &cpuset_sec_ext);
    CPU_SET(2, &cpuset_sec_ext);

    CPU_ZERO(&cpuset_k);
    CPU_SET(4, &cpuset_k);
    //CPU_SET(1, &cpuset_k);

    CPU_ZERO(&cpuset_stream0);
    CPU_SET(1, &cpuset_stream0);
    CPU_SET(2, &cpuset_stream0);

    
    CPU_ZERO(&cpuset_stream1);
    CPU_SET(1, &cpuset_stream1);

    
    CPU_ZERO(&cpuset_stream2);
    CPU_SET(2, &cpuset_stream2);


    CPU_ZERO(&cpuset_stream3);
    CPU_SET(0, &cpuset_stream3);

    CPU_ZERO(&cpuset_stream4);
    CPU_SET(3, &cpuset_stream4);

    CPU_ZERO(&cpuset_stream5);
    CPU_SET(4, &cpuset_stream5);
    
    CPU_ZERO(&cpuset_stream6);
    CPU_SET(5, &cpuset_stream6);


    //configura CPU
    CPU_ZERO(&cpusetp);
    CPU_SET(1, &cpusetp);
    CPU_ZERO(&cpusetp2);
    CPU_SET(2, &cpusetp2);


    if(pthread_attr_init(&(threads_attr[1])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[1]), sizeof(cpu_set_t), &cpuset_zero) != 0)
        printf("pthread_attr_setaffinity_np");


    if(pthread_attr_init(&(threads_attr[3])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[3]), sizeof(cpu_set_t), &cpuset_sec_ext) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[4])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[4]), sizeof(cpu_set_t), &cpuset_k) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[5])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[5]), sizeof(cpu_set_t), &cpuset_stream0) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[6])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[6]), sizeof(cpu_set_t), &cpuset_stream1) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[7])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[7]), sizeof(cpu_set_t), &cpuset_stream2) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[8])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[8]), sizeof(cpu_set_t), &cpuset_stream3) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[9])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[9]), sizeof(cpu_set_t), &cpuset_stream4) != 0)
        printf("pthread_attr_setaffinity_np");


        if(pthread_attr_init(&(threads_attr[10])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[10]), sizeof(cpu_set_t), &cpuset_stream5) != 0)
        printf("pthread_attr_setaffinity_np");

    if(pthread_attr_init(&(threads_attr[11])) != 0)
        printf("pthread_attr_init");
    if(pthread_attr_setaffinity_np(&(threads_attr[11]), sizeof(cpu_set_t), &cpuset_stream6) != 0)
        printf("pthread_attr_setaffinity_np");



    


    
    //inicializaçao dos atributos das threads
    if(pthread_attr_init(&(threads_attr[0])) != 0)
        printf("pthread_attr_init");

    //atribuição dos atributos das threads ao CPU
    if(pthread_attr_setaffinity_np(&(threads_attr[0]), sizeof(cpu_set_t), &cpusetp) != 0)
        printf("pthread_attr_setaffinity_np");

    //inicializaçao dos atributos das threads
    if(pthread_attr_init(&(threads_attr[2])) != 0)
        printf("pthread_attr_init");

    //atribuição dos atributos das threads ao CPU
    if(pthread_attr_setaffinity_np(&(threads_attr[2]), sizeof(cpu_set_t), &cpusetp2) != 0)
        printf("pthread_attr_setaffinity_np");
        
    //clock_gettime(CLOCK_MONOTONIC, &start1);
    
    pthread_create(&threads[4], &(threads_attr[4]), compute_k, args_k);
    pthread_create(&threads[3], &(threads_attr[3]), compute_sec_ext, args_sec_ext);
    pthread_create(&threads[1], &(threads_attr[1]), compute_zero_code, args_zero);
    
    
    
    

    pthread_join(threads[1], NULL);
    //printf("zero done\n");
    pthread_join(threads[4], NULL);
    //printf("k done\n");
    pthread_join(threads[3], NULL);
    ///*  */printf("sec ext done\n");
    
    pthread_create(&threads[6], &(threads_attr[6]), compute_stream_denver, args_stream1);
    pthread_create(&threads[7], &(threads_attr[7]), compute_stream_denver, args_stream2);
    pthread_create(&threads[8], &(threads_attr[8]), compute_stream_arm, args_stream3);
    pthread_create(&threads[9], &(threads_attr[9]), compute_stream_arm, args_stream4);
    pthread_create(&threads[10], &(threads_attr[10]), compute_stream_arm, args_stream5);
    pthread_create(&threads[11], &(threads_attr[11]), compute_stream_arm, args_stream6);
    
    pthread_join(threads[6], NULL);
    //printf("stream1 done\n");
    pthread_join(threads[7], NULL);
    pthread_join(threads[8], NULL);
    pthread_join(threads[9], NULL);
    pthread_join(threads[10], NULL);
    pthread_join(threads[11], NULL);
    //printf("stream2 done\n");
    //clock_gettime(CLOCK_MONOTONIC, &end3);
    //printf("writng%lf\n",(end3.tv_sec-start3.tv_sec)*1e3+(end3.tv_nsec-start3.tv_nsec)*1e-6);

    pthread_create(&threads[5], &(threads_attr[5]), compute_stream0, args_stream0);
    pthread_join(threads[5], NULL);
    /* clock_gettime(CLOCK_MONOTONIC, &end1);
    printf("%lf\n",(end1.tv_sec-start1.tv_sec)*1e3+(end1.tv_nsec-start1.tv_nsec)*1e-6);

    FILE *fp;
    fp = fopen("tempos_encoder.txt","a");
    if(fp == NULL){
        printf("Error writng file!\n");   
        exit(1);             
    }
    fprintf(fp,"%lf\n",(end1.tv_sec-start1.tv_sec)*1e3+(end1.tv_nsec-start1.tv_nsec)*1e-6);
    fclose(fp); */
    
    
    printf("stream done\n");


    
    //double StartTime=omp_get_wtime();
    /* clock_gettime(CLOCK_MONOTONIC, &start1);
    pthread_create(&threads[1], &(threads_attr[1]), compute_zero_code, args_zero);
    pthread_create(&threads[3], &(threads_attr[3]), compute_sec_ext, args_sec_ext);
    pthread_create(&threads[4], &(threads_attr[4]), compute_k, args_k);
    pthread_create(&threads[5], &(threads_attr[5]), compute_stream, args_stream);
    pthread_join(threads[1], NULL);
    printf("zero done\n");
    pthread_join(threads[3], NULL);
    printf("sec ext done\n");
    pthread_join(threads[4], NULL);
    printf("k done\n");
    pthread_join(threads[5], NULL);
    clock_gettime(CLOCK_MONOTONIC, &end1);
    printf("%lf\n",(end1.tv_sec-start1.tv_sec)*1e3+(end1.tv_nsec-start1.tv_nsec)*1e-6);
    
    printf("stream done\n"); */
    //printf("time= %lf\n", omp_get_wtime()-StartTime);
    
    

    //clock_gettime(CLOCK_MONOTONIC, &start1);
    //double StartTime=omp_get_wtime();

    //printf("time= %lf\n", omp_get_wtime()-StartTime);
    //clock_gettime(CLOCK_MONOTONIC, &end1);
    //printf("%lf\n",(end1.tv_sec-start1.tv_sec)*1e3+(end1.tv_nsec-start1.tv_nsec)*1e-6);

    //clock_gettime(CLOCK_MONOTONIC, &start1);
    //double StartTime=omp_get_wtime();
    
    //printf("time= %lf\n", omp_get_wtime()-StartTime);
    //clock_gettime(CLOCK_MONOTONIC, &end1);
    //printf("%lf\n",(end1.tv_sec-start1.tv_sec)*1e3+(end1.tv_nsec-start1.tv_nsec)*1e-6);
    


    ///aux=&k_codes[0];

    if((block_samples = (unsigned short int *)malloc(encoder_params.block_size*sizeof(unsigned short int))) == NULL){
        fprintf(stderr, "Error in allocating space to hold the block");
        return -1;
    }

    

    /* if(encoder_params.out_interleaving == BSQ){
        unsigned int x = 0, y = 0, z = 0, k=0;
        for(z = 0; z < input_params.z_size; z++){
            for(y = 0; y < input_params.y_size; y++){
                for(x = 0; x < input_params.x_size; x++){
                    //printf("residuyals=%u, ptr=%u\n",residuals[x + y*input_params.x_size + z*input_params.x_size*input_params.y_size], ptr_no_compression[k]);
                    
                    block_samples[read_samples] = residuals[x + y*input_params.x_size + z*input_params.x_size*input_params.y_size];
                    if(all_zero != 0 && block_samples[read_samples] != 0){
                        all_zero = 0;
                    }
                    read_samples++;
                    if(read_samples == encoder_params.block_size){
                        if(y == (input_params.y_size - 1) && z == (input_params.z_size - 1) && x == (input_params.x_size - 1)){
                            // trick used to signal that we are at the end of the residuals, so if there are any
                            // pending 0 blocks they must be dumped
                            //fprintf(stderr, "end of file");
                            segment_idx = SEGMENT_SIZE - 1;
                        }
                        args_ksplit->return_arg=k;
                        
                        //if(written_bytes[0]!=bytes[k] ||written_bits[0]!=  bits[k]){
                            //printf("orig_bytes=%u, orig_bits=%u, thread_bytes=%u, thread_bits=%u, k=%d\n",written_bytes[0],written_bits[0], bytes[k], bits[k], k);
                            //exit(0);
                        //}
                        k++;
                        //if(create_block(input_params, encoder_params, block_samples, all_zero, &num_zero_blocks, &segment_idx, reference_samples, compressed_stream, written_bytes, written_bits) != 0)
                            //return -1;
                        read_samples = 0;
                        all_zero = 1;
                        reference_samples = (reference_samples + 1) % encoder_params.ref_interval;
                    }
                }
            }
        }
    }
    else{
        unsigned int x = 0, y = 0, z = 0, i = 0;
        unsigned int interleaving_counter = input_params.z_size/encoder_params.out_interleaving_depth;
        if((input_params.z_size % encoder_params.out_interleaving_depth) != 0){
            interleaving_counter++;
        }
        for(y = 0; y < input_params.y_size; y++){
            for(i = 0; i < interleaving_counter; i++){
                for(x = 0; x < input_params.x_size; x++){
                    for(z = i*encoder_params.out_interleaving_depth; z < MIN((i+1)*encoder_params.out_interleaving_depth, input_params.z_size); z++){
                        block_samples[read_samples] = residuals[x + y*input_params.x_size + z*input_params.x_size*input_params.y_size];
                        if(all_zero != 0 && block_samples[read_samples] != 0){
                            all_zero = 0;
                        }
                        read_samples++;
                        if(read_samples == encoder_params.block_size){
                            if(y == (input_params.y_size - 1) && z == (input_params.z_size - 1) && x == (input_params.x_size - 1)){
                                // trick used to signal that we are at the end of the residuals, so if there are any
                                // pending 0 blocks they must be dumped
                                //fprintf(stderr, "end of file");
                                segment_idx = SEGMENT_SIZE - 1;
                            }
                            //if(create_block(input_params, encoder_params, block_samples, all_zero, &num_zero_blocks, &segment_idx, reference_samples, compressed_stream, written_bytes, written_bits) != 0)
                                //return -1;
                            read_samples = 0;
                            all_zero = 1;
                            reference_samples = (reference_samples + 1) % encoder_params.ref_interval;
                        }
                    }
                }
            }
        }
    }
 */
    //printf("bytes=%u, bits=%u\n", written_bytes[0],written_bits[0]);

    
    // Now we have to check if the number of samples was a multiple of the block size;
    // if not we need to add zeros to the block and perform the compression.
    if(read_samples < encoder_params.block_size && read_samples > 0){
        if(all_zero == 0){
            int i = 0;
            for(i = read_samples; i < encoder_params.block_size; i++){
                block_samples[i] = 0;
            }
        }else{
            num_zero_blocks++;
        }
        if(num_zero_blocks > 0){
            printf("in\n");
            zero_block_code(input_params, encoder_params, num_zero_blocks, compressed_stream, written_bytes, written_bits, 0);
        }
        if(all_zero == 0){
            printf("in\n");

            compute_block_code(input_params, encoder_params, block_samples, compressed_stream, written_bytes, written_bits);
        }
    }
    if(block_samples != NULL){
        free(block_samples);
    }
    
    sem_destroy(&sem_zero);
    sem_destroy(&sem_sec_ext);
    sem_destroy(&sem_k);

    /* err=cudaFreeHost(k_codes);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to free the mpr from the host (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }


    err=cudaFreeHost(len_k);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to free the mpr from the host (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    } */


    if(ptr_zero_block != NULL)
        free(ptr_zero_block);;
    if(ptr_sec_ext != NULL)
        free(ptr_sec_ext);
    if(ptr_k != NULL)
        free(ptr_k);
    if(ptr_stream != NULL)
        free(ptr_stream);


    if(len_sec_ext != NULL)
        free(len_sec_ext);
    if(len_k != NULL)
        free(len_k); 
    if(zero_codes != NULL)
        free(zero_codes);
    if(sec_ext_codes != NULL)
        free(sec_ext_codes);
    if(k_codes != NULL)
        free(k_codes); 

    if(stream1 != NULL)
        free(stream1);
    if(stream2 != NULL)
        free(stream2);

    if(written_bytes1 != NULL)
        free(written_bytes1);
    if(written_bits1 != NULL)
        free(written_bits1);
    if(written_bytes2 != NULL)
        free(written_bytes2);
    if(written_bits2 != NULL)
        free(written_bits2);


        
    if(args_second != NULL)
        free(args_second);
    if(args_ksplit != NULL)
        free(args_ksplit);
    if(args_zero != NULL)
        free(args_zero);
    if(args_sec_ext != NULL)
        free(args_sec_ext);
    if(args_k != NULL)
        free(args_k);
    if(args_stream0 != NULL)
        free(args_stream0);
    if(args_stream1 != NULL)
        free(args_stream1);
    if(args_stream2 != NULL)
        free(args_stream2);


    munlockall();//desbloqueia memoria
    return 0;
}

/******************************************************
* END Block Adaptive Routines
*******************************************************/



///Main function for the entropy encoding of a given input file; while it works for any input file,
///it is though to be used when the input file encodes the residuals of each pixel of an image after
///the lossless compression step
///@param input_params describe the image whose residuals are contained in the input file
///@param encoder_params set of options determining the behavior of the encoder
///@param inputFile file containing the information to be compressed 
///@param outputFile file where the compressed information will be stored
///@return the number of bytes which compose the compressed stream, a negative value if an error
///occurred
extern "C" int encode(input_feature_t input_params, encoder_config_t encoder_params, predictor_config_t predictor_params, 
    unsigned short int * residuals, char outputFile[128]){
    // The function is pretty simple; it mainly simply parses the input files,
    // and calls the encode_core routine. After the encoding has ended it writes the
    // result to the output file.
    // all memory allocation/de-allocation takes place inside this routine
    unsigned long * compressed_stream = NULL;
    int encoding_outcome = 0, write_result = 0;
    //unsigned int num_padding_bits = 0;
    unsigned int written_bytes = 0, written_bits = 0;
    FILE * outFile = NULL;

    // Note how the compressed stream shall never be greater than the original size of the
    // residuals
    compressed_stream = (unsigned long *)malloc((((input_params.dyn_range + 8)/8)*input_params.x_size*input_params.y_size*input_params.z_size));
    if(compressed_stream == NULL){
        fprintf(stderr, "Error in the allocation of the compressed stream\n\n");
        return -1;
    }
    memset(compressed_stream, 0, (((input_params.dyn_range + 7)/8)*input_params.x_size*input_params.y_size*input_params.z_size));

    // First of all we need to write the headers to the file
    //clock_gettime(CLOCK_MONOTONIC, &start1);
    //create_header(&written_bytes, &written_bits, compressed_stream, input_params, predictor_params, encoder_params);
    //clock_gettime(CLOCK_MONOTONIC, &end1);
    //printf("%lf\n",(end1.tv_sec-start1.tv_sec)*1e3+(end1.tv_nsec-start1.tv_nsec)*1e-6);

    // Finally I can perform the encoding
    if(encoder_params.encoding_method == SAMPLE){
        encoding_outcome = encode_sampleadaptive(input_params, predictor_params, encoder_params, residuals, compressed_stream, &written_bytes, &written_bits);
    }else{
        encoding_outcome = encode_block(input_params, predictor_params, encoder_params, residuals, compressed_stream, &written_bytes, &written_bits);
    }
    if(encoding_outcome < 0){
        fprintf(stderr, "Error in encodying the residuals\n\n");
        return -1;
    }
    
    // Compression has finished; I fill up the compressed stream bits to pad it to
    // word length and deallocate memory
    /*num_padding_bits = encoder_params.out_wordsize*8 - ((written_bytes*8 + written_bits) % (encoder_params.out_wordsize*8));
    if(num_padding_bits < encoder_params.out_wordsize*8 && num_padding_bits > 0){
        printf("in\n");

        bitStream_store_constant(compressed_stream, &written_bytes, &written_bits, num_padding_bits, 0);
    }*/

    // and saving the results on the output file
    if((outFile = fopen(outputFile, "wb")) == NULL){
        fprintf(stderr, "Error in creating file %s for writing the compression result\n\n", outputFile);
        return -1;
    }
    //printf("%hu %hu %hu %hu\n", compressed_stream[0], compressed_stream[1], compressed_stream[2], compressed_stream[3]);
    for(int i=0; i<=written_bytes; i++){
        unsigned long aux=compressed_stream[i] &  0xff00000000000000;
        unsigned long aux2=compressed_stream[i] & 0x00ff000000000000;
        unsigned long aux3=compressed_stream[i] & 0x0000ff0000000000;
        unsigned long aux4=compressed_stream[i] & 0x000000ff00000000;
        unsigned long aux5=compressed_stream[i] & 0x00000000ff000000;
        unsigned long aux6=compressed_stream[i] & 0x0000000000ff0000;
        unsigned long aux7=compressed_stream[i] & 0x000000000000ff00;
        aux=aux>>56;
        aux2=aux2>>40;
        aux3=aux3>>24;
        aux4=aux4>>8;
        aux5=aux5<<8;
        aux6=aux6<<24;
        aux7=aux7<<40;
        compressed_stream[i]=compressed_stream[i]<<56;
        compressed_stream[i]=compressed_stream[i]|aux;
        compressed_stream[i]=compressed_stream[i]|aux2;
        compressed_stream[i]=compressed_stream[i]|aux3;
        compressed_stream[i]=compressed_stream[i]|aux4;
        compressed_stream[i]=compressed_stream[i]|aux5;
        compressed_stream[i]=compressed_stream[i]|aux6;
        compressed_stream[i]=compressed_stream[i]|aux7;

    }
    

    //printf("%x\n", compressed_stream[written_bytes-1]);
    //write_result = fwrite(compressed_stream, 2, written_bytes, outFile);
    write_result = fwrite(compressed_stream, sizeof(unsigned long), written_bytes+1, outFile);
    if(write_result != written_bytes+1){
        fprintf(stderr, "Error in writing compressed stream to %s: only %d bytes out of %d written\n\n", outputFile, write_result,written_bytes+1);
        return -1;        
    }
    fclose(outFile);

    if(compressed_stream != NULL)
        free(compressed_stream);
    return written_bytes;
}