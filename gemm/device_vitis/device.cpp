#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"

using namespace xf::blas;

typedef int t_TypeInt;
typedef double BLAS_dataType;

#define BLAS_parEntries 16
#define BLAS_k 100

extern "C" {
void gemm_kernel(double * p_A, double * p_B, double * result, unsigned int p_m, unsigned int p_n, unsigned int p_k) {
    #pragma HLS INTERFACE m_axi port=p_A offset=slave bundle=port1
    #pragma HLS INTERFACE m_axi port=p_B offset=slave bundle=port2
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=p_A bundle=control
    #pragma HLS INTERFACE s_axilite port=p_B bundle=control
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=p_m bundle=control
    #pragma HLS INTERFACE s_axilite port=p_n bundle=control
    #pragma HLS INTERFACE s_axilite port=p_k bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

	  hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strA;
    hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strB;
    hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strC;
#pragma HLS DATAFLOW
    gemmMatAMover<BLAS_dataType, BLAS_parEntries>(p_A, p_m, p_n, p_k, l_strA);
    gemmMatBMover<BLAS_dataType, BLAS_parEntries>(p_B, p_m, p_n, p_k, l_strB);
    gemm<BLAS_dataType, BLAS_k, BLAS_parEntries>(p_m, p_n, p_k, l_strA, l_strB, l_strC);
    writeStream2Vec<BLAS_dataType, BLAS_parEntries>(l_strC, p_m * p_n, result);
}

}
