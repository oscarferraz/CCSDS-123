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

#ifndef ENTROPY_ENCODER_H
#define ENTROPY_ENCODER_H


#include <cuda_runtime.h> 

#include "utils.h"
#include "predictor.h"

typedef enum{SAMPLE, BLOCK} encoder_t;

///Type representing the configuration of the encoder algorithm
typedef struct encoder_config{
    unsigned int u_max;
    unsigned int y_star;
    unsigned int y_0;
    unsigned int k;
    unsigned int * k_init;
    interleaving_t out_interleaving;
    unsigned int out_interleaving_depth;
    unsigned int out_wordsize;
    encoder_t encoding_method;
    unsigned char block_size;
    unsigned char restricted;
    unsigned int ref_interval;
} encoder_config_t;



struct thread_args_second {
    encoder_config_t encoder_params;
    unsigned short int * block_samples;
    unsigned int *second_extension_values;
     int return_arg;
  };

 


  struct thread_args_split {
    input_feature_t input_params;
    encoder_config_t encoder_params;
    unsigned short int * block_samples;
    int * k_split;
    unsigned int return_arg;
  };

  struct thread_args_zero {
    unsigned short int * ptr_zero_block;
    unsigned char * zero_codes;
    unsigned int size;
  };

  struct thread_args_sec_ext {
    unsigned short int * ptr_sec_ext;
    unsigned int * len_sec_ext;
    unsigned int * sec_ext_codes;
    unsigned int size;
  };

  struct thread_args_k {
    unsigned short int * ptr_k;
    unsigned int * len_k;
    int * k_codes;
    unsigned int size;
  };

  struct thread_args_stream {
    unsigned short int * ptr_stream;
    unsigned char * stream_zero_codes;
    unsigned long * compressed_stream;
    unsigned int * written_bytes;
    unsigned int * written_bits;
    unsigned int * stream_len_sec_ext;
    unsigned int * stream_len_k;
    int * stream_k_codes;
    unsigned int * stream_sec_ext_codes;
    unsigned int n_blocks;
    unsigned char id;

  };

  struct thread_args_stream_conc {
    unsigned long * compressed_stream;
    unsigned int * written_bytes;
    unsigned int * written_bits;
    unsigned long * stream1;
    unsigned int * written_bytes1;
    unsigned int * written_bits1;
    unsigned long * stream2;
    unsigned int * written_bytes2;
    unsigned int * written_bits2;
    unsigned long * stream3;
    unsigned int * written_bytes3;
    unsigned int * written_bits3;
    unsigned long * stream4;
    unsigned int * written_bytes4;
    unsigned int * written_bits4;
    unsigned long * stream5;
    unsigned int * written_bytes5;
    unsigned int * written_bits5;
    unsigned long * stream6;
    unsigned int * written_bytes6;
    unsigned int * written_bits6;
    unsigned long * stream7;
    unsigned int * written_bytes7;
    unsigned int * written_bits7;
    unsigned long * stream8;
    unsigned int * written_bytes8;
    unsigned int * written_bits8;
    input_feature_t input_params;
    predictor_config_t predictor_params;
    encoder_config_t encoder_params;
  };

    struct thread_args_sample {
    unsigned long * compressed_stream;
    unsigned int * written_bytes;
    unsigned int * written_bits;
    unsigned int * counter;
    unsigned int * accumulator;
    unsigned short int * residuals;
    unsigned short bands;
    input_feature_t input_params;
  };


///Main function for the entropy encoding of a given input file; while it works for any input file,
///it is though to be used when the input file encodes the residuals of each pixel of an image after
///the lossless compression step
///@param input_params describe the image whose residuals are contained in the input file
///@param encoder_params set of options determining the behavior of the encoder
///@param inputFile file containing the information to be compressed
///@param outputFile file where the compressed information will be stored
///@return the number of bytes which compose the compressed stream, a negative value if an error
///occurred
__global__ void GPU_compute_k_split_enc(unsigned short *d_mpr, unsigned int * d_len, int * d_code);

__global__ void GPU_compute_sample(unsigned short *d_mpr, unsigned long * d_stream, unsigned int * d_bytes, unsigned int * d_bits, unsigned int dim_x, unsigned int dim_y);

__device__ void gpu_bitStream_store_constant(unsigned long * compressed_stream, unsigned int * written_bytes,
        unsigned int * written_bits, unsigned int num_bits_to_write, unsigned char bit_to_repeat, unsigned int offset, unsigned int z);

__device__ void gpu_bitStream_store(unsigned long * compressed_stream, unsigned int * written_bytes,
        unsigned int * written_bits, unsigned int num_bits_to_write, unsigned long bits_to_write, unsigned int offset, unsigned int z);


extern int encode(input_feature_t input_params, encoder_config_t encoder_params, predictor_config_t predictor_params, unsigned short int * residuals, char outputFile[128]);




#endif
